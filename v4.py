import cv2
import numpy as np
import matplotlib.pyplot as plt
from lib.RecentFrameTracker import RecentFrameTracker
from lib.ChartDataLogger import ChartDataLogger



# PNG generation
def generate_and_save_plot_of_raw_camera_values(avg_positions, file_name):
    if not avg_positions:
        print("No average positions logged.")
        return
    
    valid_indices = [i for i, pos in enumerate(avg_positions)]
    if not valid_indices:
        print("No valid data positions logged.")
        return

    first_valid_index = valid_indices[0]
    last_valid_index = valid_indices[-1]

    time_steps = list(range(len(avg_positions)))
    avg_x_positions = [pos[0] for pos in avg_positions]
    avg_y_positions = [pos[1] for pos in avg_positions]
    avg_r_positions = [pos[2] for pos in avg_positions]

    plt.figure(dpi = 300)
    plt.plot(time_steps, avg_x_positions, label='X Screen Position', marker='o', antialiased=False)
    plt.plot(time_steps, avg_y_positions, label='Y Screen Position', marker='x', antialiased=False)
    plt.plot(time_steps, avg_r_positions, label='Radius', marker='^', antialiased=False)

    plt.title('X, Y Positions and Radius Over Time')
    plt.xlabel('Time (Frame Index)')
    plt.ylabel('Average Value')
    plt.legend()
    plt.grid(True)

    # Changing the following line would ruin logic in the video of the graph, and adjust the limits on x axis
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.xlim(first_valid_index, last_valid_index)

    plt.savefig(file_name)
    print(f"Plot saved as {file_name}")

# Adds the graph to the video
def overlay_graph_on_frame(frame, graph, frame_number, total_frames):
    graph_resized = cv2.resize(graph, (frame.shape[1] // 3, frame.shape[0] // 3))
    x_offset = frame.shape[1] - graph_resized.shape[1]
    y_offset = frame.shape[0] - graph_resized.shape[0]
    frame[y_offset:y_offset+graph_resized.shape[0], x_offset:x_offset+graph_resized.shape[1]] = graph_resized

    # Assuming margins are about 10% of the graph image on each side
    buffer_percentage_horizontal = 0.10
    buffer_percentage_vertical = 0.10
    buffer_size_horizontal = int(buffer_percentage_horizontal * graph_resized.shape[1])
    buffer_size_vertical = int(buffer_percentage_vertical * graph_resized.shape[0])

    # Adjust the position to start at the beginning of the actual plot area
    adjusted_graph_width = graph_resized.shape[1] - 2 * buffer_size_horizontal
    position = int((frame_number / total_frames) * adjusted_graph_width) + x_offset + buffer_size_horizontal

    # Draw the bar
    bar_vertical_start = y_offset + buffer_size_vertical
    bar_vertical_end = y_offset + graph_resized.shape[0] - buffer_size_vertical
    cv2.line(frame, (position, bar_vertical_start), (position, bar_vertical_end), (0, 255, 0), 2)

    return frame

# As
def add_graph_to_video(video_path, graph_path, output_video_path):
    graph_image = cv2.imread(graph_path)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        frame_with_graph = overlay_graph_on_frame(frame, graph_image, i, total_frames)
        out.write(frame_with_graph)

    cap.release()
    out.release()

def track_ball_grayscale_and_position(save_video):
    cap = cv2.VideoCapture(0)
    window_name = 'Ball Tracking'
    cv2.namedWindow(window_name)
    FRAME_RATE = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera Frame Rate: {FRAME_RATE}")
    base_root = "Output"

    recent_frame_tracker = RecentFrameTracker(10)
    
    if(save_video):
        chart_data_logger = ChartDataLogger()
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(base_root + '/video.avi', fourcc, 20, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([30, 255, 255])
        green_lower = np.array([35, 50, 50])
        green_upper = np.array([90, 255, 255])

        mask_yellow = cv2.inRange(hsv_frame, yellow_lower, yellow_upper)
        mask_green = cv2.inRange(hsv_frame, green_lower, green_upper)

        contours, _ = cv2.findContours(mask_green.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None

        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)

            if radius > 10:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), -1)
                center = (int(x), int(y))

                recent_frame_tracker.log_position(x, y, radius)
                chart_data_logger.log_position(x, y, radius)

                avg_position = recent_frame_tracker.get_avg()
                xPos = round(avg_position[0])
                yPos = round(avg_position[1])
                r = round(avg_position[2])
        
                if avg_position:
                    avg_position_text = "X: " + str(xPos) + ", Y: " + str(yPos) + ", r: "+ str(r)
                    cv2.putText(frame, avg_position_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(window_name, frame)

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        if cv2.waitKey(1) & 0xFF == ord('p'):
            if(save_video):
                generate_and_save_plot_of_raw_camera_values(chart_data_logger.get_all_positions(), base_root + "/complete_tracking_data.png")
                generate_and_save_plot_of_raw_camera_values(chart_data_logger.get_positions_local_avg(10), base_root + "/with average.png")
                break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        out.write(frame) # Add frame to video


    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    if(save_video):
        add_graph_to_video(base_root+'/video.avi', base_root+'/with average.png', base_root+'/final_output.avi')
        add_graph_to_video(base_root+'/video.avi', base_root+'/complete_tracking_data.png', base_root+'/final_output_no_smoothing.avi')




track_ball_grayscale_and_position(True)