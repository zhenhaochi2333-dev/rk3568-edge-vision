#include "edgevision/rtsp_streamer.hpp"
#include "edgevision/logger.hpp"

#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <gst/rtsp-server/rtsp-server.h>

#if defined(EDGEVISION_WITH_X11_THREADS) && EDGEVISION_WITH_X11_THREADS
#include <X11/Xlib.h>
#endif

#include <cstring>
#include <stdexcept>
#include <utility>

namespace edgevision {

namespace {

constexpr int kWidth = 1280;
constexpr int kHeight = 720;
constexpr int kFps = 15;
constexpr int kBitrate = 6000000;

GstClockTime frame_duration()
{
    return gst_util_uint64_scale_int(GST_SECOND, 1, kFps);
}

}  // namespace

RtspStreamer::RtspStreamer(int port, std::string advertised_host)
    : port_(port),
      url_("rtsp://" + std::move(advertised_host) + ":" + std::to_string(port) + "/live"),
      pipeline_(make_pipeline())
{
    if (port_ < 1 || port_ > 65535) {
        throw std::runtime_error("RTSP port must be within [1,65535]");
    }
}

RtspStreamer::~RtspStreamer()
{
    stop();
}

std::string RtspStreamer::make_pipeline()
{
    return "( appsrc name=src is-live=true format=time do-timestamp=false "
           "block=false max-bytes=2764800 "
           "caps=\"video/x-raw,format=BGR,width=1280,height=720,framerate=15/1\" "
           "! videoconvert ! video/x-raw,format=NV12,width=1280,height=720,framerate=15/1 "
           "! mpph264enc bps=6000000 gop=15 header-mode=each-idr "
           "! h264parse ! rtph264pay name=pay0 pt=96 config-interval=1 )";
}

void RtspStreamer::start()
{
    std::unique_lock<std::mutex> lock(mutex_);
    if (running_) {
        return;
    }
    if (server_thread_.joinable()) {
        lock.unlock();
        server_thread_.join();
        lock.lock();
    }
    ready_ = false;
    failed_ = false;
    stop_requested_ = false;
    error_message_.clear();
#if defined(EDGEVISION_WITH_X11_THREADS) && EDGEVISION_WITH_X11_THREADS
    // OpenCV HighGUI and the capture/RTSP worker threads share the board's
    // X11 connection. Enable Xlib's thread support before the first window
    // is created.
    XInitThreads();
#endif
    // Initialize GStreamer on the application thread. The board also uses
    // OpenCV HighGUI, so doing this from the RTSP worker can race X11/XCB
    // initialization.
    gst_init(nullptr, nullptr);
    server_thread_ = std::thread(&RtspStreamer::run_server, this);
    condition_.wait(lock, [this] { return ready_ || failed_; });
    if (failed_) {
        const std::string message = error_message_;
        lock.unlock();
        if (server_thread_.joinable()) {
            server_thread_.join();
        }
        throw std::runtime_error(message);
    }
}

void RtspStreamer::stop()
{
    _GMainLoop* loop = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_requested_ = true;
        loop = loop_;
    }
    if (loop != nullptr) {
        g_main_loop_quit(reinterpret_cast<GMainLoop*>(loop));
    }
    condition_.notify_all();
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
}

bool RtspStreamer::is_running() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return running_;
}

void RtspStreamer::publish(const cv::Mat& annotated_bgr)
{
    if (annotated_bgr.empty() || annotated_bgr.cols != kWidth ||
        annotated_bgr.rows != kHeight || annotated_bgr.type() != CV_8UC3) {
        throw std::runtime_error("RTSP frame must be 1280x720 BGR CV_8UC3");
    }

    GstBuffer* buffer = gst_buffer_new_allocate(nullptr,
                                                 annotated_bgr.total() * annotated_bgr.elemSize(),
                                                 nullptr);
    if (buffer == nullptr) {
        throw std::runtime_error("cannot allocate RTSP frame buffer");
    }

    GstMapInfo map{};
    if (!gst_buffer_map(buffer, &map, GST_MAP_WRITE)) {
        gst_buffer_unref(buffer);
        throw std::runtime_error("cannot map RTSP frame buffer");
    }
    std::memcpy(map.data, annotated_bgr.data, map.size);
    gst_buffer_unmap(buffer, &map);

    std::lock_guard<std::mutex> lock(mutex_);
    if (latest_buffer_ != nullptr) {
        gst_buffer_unref(reinterpret_cast<GstBuffer*>(latest_buffer_));
    }
    latest_buffer_ = reinterpret_cast<_GstBuffer*>(buffer);
    condition_.notify_all();
}

void RtspStreamer::on_media_configure(_GstRTSPMediaFactory*,
                                      _GstRTSPMedia* media,
                                      void* user_data)
{
    static_cast<RtspStreamer*>(user_data)->configure_media(media);
}

void RtspStreamer::configure_media(_GstRTSPMedia* media)
{
    GstRTSPMedia* rtsp_media = reinterpret_cast<GstRTSPMedia*>(media);
    // Keep the appsrc/MPP encoder alive when a client disappears.  This is
    // required for a subsequent client to reuse the same prepared media
    // instead of trying to re-preroll the Rockchip encoder from scratch.
    gst_rtsp_media_set_reusable(rtsp_media, TRUE);
    gst_rtsp_media_set_stop_on_disconnect(rtsp_media, FALSE);
    gst_rtsp_media_set_suspend_mode(rtsp_media, GST_RTSP_SUSPEND_MODE_NONE);
    gst_rtsp_media_set_eos_shutdown(rtsp_media, FALSE);

    GstElement* element = gst_rtsp_media_get_element(rtsp_media);
    if (element == nullptr) {
        log_warn("RTSP media has no pipeline element");
        return;
    }

    GstElement* appsrc = gst_bin_get_by_name_recurse_up(GST_BIN(element), "src");
    if (appsrc == nullptr) {
        gst_object_unref(element);
        log_warn("RTSP media pipeline has no appsrc named src");
        return;
    }

    GstAppSrcCallbacks callbacks{};
    callbacks.need_data = &RtspStreamer::on_need_data;
    gst_app_src_set_callbacks(GST_APP_SRC(appsrc), &callbacks, this, nullptr);
    g_object_set(G_OBJECT(appsrc),
                 "is-live", TRUE,
                 "stream-type", GST_APP_STREAM_TYPE_STREAM,
                 "format", GST_FORMAT_TIME,
                 "do-timestamp", FALSE,
                 "block", FALSE,
                 "max-bytes", static_cast<guint64>(kWidth * kHeight * 3),
                 nullptr);
    // RTSP DESCRIBE needs the payloader caps while the media is being
    // prepared. Seed appsrc with the newest annotated frame before the
    // media pipeline is asked to produce its SDP.
    push_latest(reinterpret_cast<_GstAppSrc*>(appsrc));
    gst_object_unref(appsrc);
    gst_object_unref(element);
}

void RtspStreamer::on_need_data(_GstAppSrc* appsrc, unsigned int, void* user_data)
{
    static_cast<RtspStreamer*>(user_data)->push_latest(appsrc);
}

void RtspStreamer::push_latest(_GstAppSrc* appsrc)
{
    GstBuffer* buffer = nullptr;
    GstClockTime pts = GST_CLOCK_TIME_NONE;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (latest_buffer_ == nullptr || stop_requested_) {
            return;
        }
        buffer = gst_buffer_copy(reinterpret_cast<GstBuffer*>(latest_buffer_));
        pts = next_pts_;
        next_pts_ += frame_duration();
    }

    if (buffer == nullptr) {
        return;
    }
    GST_BUFFER_PTS(buffer) = pts;
    GST_BUFFER_DTS(buffer) = pts;
    GST_BUFFER_DURATION(buffer) = frame_duration();
    const GstFlowReturn result =
        gst_app_src_push_buffer(reinterpret_cast<GstAppSrc*>(appsrc), buffer);
    if (result != GST_FLOW_OK && result != GST_FLOW_FLUSHING) {
        log_warn("RTSP appsrc push returned flow=" + std::to_string(result));
    }
}

void RtspStreamer::run_server()
{
    GMainContext* context = nullptr;
    GMainLoop* loop = nullptr;
    GstRTSPServer* server = nullptr;
    GstRTSPMediaFactory* factory = nullptr;
    guint source_id = 0U;

    try {
        // Keep RTSP's GLib dispatch separate from OpenCV HighGUI's default
        // context. Both are active concurrently in smooth-preview mode.
        context = g_main_context_new();
        loop = g_main_loop_new(context, FALSE);
        server = gst_rtsp_server_new();
        factory = gst_rtsp_media_factory_new();
        if (context == nullptr || loop == nullptr || server == nullptr || factory == nullptr) {
            throw std::runtime_error("cannot allocate RTSP server objects");
        }

        gst_rtsp_server_set_address(server, "0.0.0.0");
        gst_rtsp_server_set_service(server, std::to_string(port_).c_str());
        gst_rtsp_media_factory_set_launch(factory, pipeline_.c_str());
        // Keep one encoder/appsrc pipeline alive across client disconnects.
        // Recreating the Rockchip encoder on every RTSP reconnect can leave
        // the new media instance unprepared and produce not-negotiated.
        gst_rtsp_media_factory_set_shared(factory, TRUE);
        gst_rtsp_media_factory_set_stop_on_disconnect(factory, FALSE);
        gst_rtsp_media_factory_set_suspend_mode(factory, GST_RTSP_SUSPEND_MODE_NONE);
        gst_rtsp_media_factory_set_eos_shutdown(factory, FALSE);
        g_signal_connect(factory, "media-configure",
                         G_CALLBACK(&RtspStreamer::on_media_configure), this);

        GstRTSPMountPoints* mounts = gst_rtsp_server_get_mount_points(server);
        gst_rtsp_mount_points_add_factory(mounts, "/live", factory);
        g_object_unref(mounts);
        factory = nullptr;

        source_id = gst_rtsp_server_attach(server, context);
        g_main_context_unref(context);
        context = nullptr;
        if (source_id == 0U) {
            throw std::runtime_error("cannot attach RTSP server to its main context");
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            loop_ = reinterpret_cast<_GMainLoop*>(loop);
            running_ = true;
            ready_ = true;
            condition_.notify_all();
        }
        g_main_loop_run(loop);
    } catch (const std::exception& error) {
        std::lock_guard<std::mutex> lock(mutex_);
        failed_ = true;
        ready_ = true;
        error_message_ = error.what();
        condition_.notify_all();
    }

    if (source_id != 0U) {
        g_source_remove(source_id);
    }
    if (context != nullptr) {
        g_main_context_unref(context);
    }
    if (factory != nullptr) {
        g_object_unref(factory);
    }
    if (server != nullptr) {
        g_object_unref(server);
    }
    if (loop != nullptr) {
        g_main_loop_unref(loop);
    }

    std::lock_guard<std::mutex> lock(mutex_);
    loop_ = nullptr;
    running_ = false;
    if (latest_buffer_ != nullptr) {
        gst_buffer_unref(reinterpret_cast<GstBuffer*>(latest_buffer_));
        latest_buffer_ = nullptr;
    }
    condition_.notify_all();
}

}  // namespace edgevision
