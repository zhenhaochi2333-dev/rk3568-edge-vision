/* 已标注 BGR 帧 -> GStreamer appsrc -> NV12 -> Rockchip MPP H.264 -> RTP/RTSP。
 * 此输出链与输入侧的 JPEG/TCP 是两个独立协议和编码方向。
 */
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
    // 用 GStreamer 时钟刻度表示 1/15 秒，PTS 在单一媒体时间轴上按该间隔递增。
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

// appsrc 接受 1280x720 BGR；videoconvert 交付 NV12，再由 mpph264enc 编码。
std::string RtspStreamer::make_pipeline()
{
    // appsrc 输入紧凑 BGR；videoconvert 转成硬件编码器需要的 NV12，
    // MPP 输出 H.264，解析器和 payloader 再生成 RTSP/RTP 传输所需的负载。
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
    // 上次线程即使已退出，仍需 join 才能回收 thread 对象后重新赋值。
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
    // 同步等到工作线程报告初始化结果，避免调用方在服务未就绪时开始推帧。
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
    // 在锁内取得指针快照，锁外通知 GLib 退出，避免持锁等待服务线程清理。
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

// 从调用者 Mat 复制一帧到自有 GstBuffer；替换旧帧后 RTSP 客户端读取不依赖 Mat 寿命。
void RtspStreamer::publish(const cv::Mat& annotated_bgr)
{
    if (annotated_bgr.empty() || annotated_bgr.cols != kWidth ||
        annotated_bgr.rows != kHeight || annotated_bgr.type() != CV_8UC3) {
        throw std::runtime_error("RTSP frame must be 1280x720 BGR CV_8UC3");
    }

    // 分配 GStreamer 自有存储，不能仅包装 annotated_bgr.data，因为调用者返回后
    // 可能立刻复用或销毁 Mat；复制使 appsrc 的缓冲寿命与输入 Mat 分离。
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
    // 当前生产调用方交付连续 BGR canvas；此处按 total*elemSize 一次复制。
    // publish 未检查 Mat::isContinuous，带行 stride 的 ROI Mat 不满足这一隐含前提。
    std::memcpy(map.data, annotated_bgr.data, map.size);
    gst_buffer_unmap(buffer, &map);

    std::lock_guard<std::mutex> lock(mutex_);
    // 成员持有当前帧的一份引用；替换时释放旧引用，但已被回调复制的帧仍独立有效。
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
    // 客户端断开后复用媒体对象，避免重建硬件编码器造成重连预热失败。
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

    // 回调把 user_data 指向当前实例；该实例在 stop() join 服务线程前一直存活。
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

// need-data 回调复制当前最新缓冲，并按 15 FPS 固定步长标记 PTS/DTS。
void RtspStreamer::push_latest(_GstAppSrc* appsrc)
{
    GstBuffer* buffer = nullptr;
    GstClockTime pts = GST_CLOCK_TIME_NONE;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (latest_buffer_ == nullptr || stop_requested_) {
            return;
        }
        // 在互斥区复制 GstBuffer 及其数据引用，之后即使 publish 替换最新帧，
        // 本次推送仍有独立引用；GStreamer 可共享底层数据，避免重复整帧 memcpy。
        buffer = gst_buffer_copy(reinterpret_cast<GstBuffer*>(latest_buffer_));
        pts = next_pts_;
        // 时间戳按实际被请求并推送的帧增长；没有请求时不会凭空补齐漏掉的帧。
        next_pts_ += frame_duration();
    }

    if (buffer == nullptr) {
        return;
    }
    // 时间戳属于推送给 appsrc 的副本；gst_app_src_push_buffer 接管该副本所有权。
    GST_BUFFER_PTS(buffer) = pts;
    GST_BUFFER_DTS(buffer) = pts;
    GST_BUFFER_DURATION(buffer) = frame_duration();
    // gst_app_src_push_buffer 接管 buffer 所有权；返回后本函数不再释放或访问它。
    const GstFlowReturn result =
        gst_app_src_push_buffer(reinterpret_cast<GstAppSrc*>(appsrc), buffer);
    if (result != GST_FLOW_OK && result != GST_FLOW_FLUSHING) {
        log_warn("RTSP appsrc push returned flow=" + std::to_string(result));
    }
}

// GLib 主循环在服务线程处理 RTSP 会话；停止时退出循环并释放 GStreamer 引用。
void RtspStreamer::run_server()
{
    // GLib、RTSP server 和 factory 在该线程创建、驱动并释放；其他线程只经 mutex_ 访问状态。
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

        // attach 把监听 socket 注册到专用 context；只有运行对应 loop 才会处理连接事件。
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

    // 先移除挂载的监听源，再释放 server/loop，确保清理期间没有新回调访问这些对象。
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
