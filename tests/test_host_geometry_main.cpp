void run_display_composer_tests();
void run_geometry_tests();
void run_iou_tracker_tests();
void run_region_monitor_tests();
void run_semantic_stabilizer_tests();

int main()
{
    run_geometry_tests();
    run_iou_tracker_tests();
    run_region_monitor_tests();
    run_semantic_stabilizer_tests();
    run_display_composer_tests();
    return 0;
}
