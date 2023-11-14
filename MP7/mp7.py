import cv2
import numpy as np


def select_bounding_box(frame_path):
    frame = cv2.imread(frame_path)
    if frame is None:
        raise FileNotFoundError(f"Unable to open the image file: {frame_path}")
    
    points = []
    def click_event(event, x, y, flags, params):
        nonlocal points
        display = frame.copy()

        if event == cv2.EVENT_LBUTTONDOWN:
            points = [(x, y)]
        
        elif event == cv2.EVENT_LBUTTONUP:
            points.append((x, y))
            cv2.rectangle(display, points[0], points[1], (255, 0, 0), 2)
            cv2.imshow("Frame", display)

    cv2.imshow('Frame', frame)
    cv2.setMouseCallback('Frame', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) != 2:
        raise Exception("Two points were not selected.")
    
    x1, y1 = points[0]
    x2, y2 = points[1]
    bbox = (min(x1, x2), min(y1, y2), abs(x2-x1), abs(y2-y1))  # Create bounding box tuple
    return bbox


def track_object(frame_path_pattern, bbox, total_frames, method):
    x, y, w, h = bbox
    
    template = cv2.imread(frame_path_pattern.format(1), cv2.IMREAD_GRAYSCALE)[y:y+h, x:x+w]

    tracking_results = []

    for frame_idx in range(1, total_frames + 1):
        frame_path = frame_path_pattern.format(frame_idx)
        frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        
        if frame is None:
            break

        # exhaustive search
        best_match_score = float('inf') if method == 'SSD' else 0
        best_match_location = (x, y)
        for i in range(0, frame.shape[0] - h):
            for j in range(0, frame.shape[1] - w):
                search_window = frame[i:i+h, j:j+w]
                if method == 'SSD':
                    score = ssd_match(template, search_window)
                    if score < best_match_score:
                        best_match_score = score
                        best_match_location = (j, i)
                elif method == 'CC':
                    score = cc_match(template, search_window)
                    if score > best_match_score:
                        best_match_score = score
                        best_match_location = (j, i)
                elif method == 'NCC':
                    score = ncc_match(template, search_window)
                    if score > best_match_score:
                        best_match_score = score
                        best_match_location = (j, i)

        # Update the tracking window and template
        x, y = best_match_location
        template = frame[y:y+h, x:x+w]

        # Append the result as a tuple of top-left and bottom-right coordinates
        tracking_results.append(((x, y), (x + w, y + h)))

    return tracking_results


def create_video(input_path_pattern, output_video_path, tracking_results, total_frames):
    # Initialize video writer
    first_frame = cv2.imread(input_path_pattern.format(1))
    if first_frame is None:
        raise FileNotFoundError("Unable to open the first frame for the video writer.")
    
    height, width, layers = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 format
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (width, height))

    for frame_idx in range(1, total_frames + 1):
        frame_path = input_path_pattern.format(frame_idx)
        frame = cv2.imread(frame_path)
        if frame is None:
            break

        # Draw the tracking result
        top_left, bottom_right = tracking_results[frame_idx - 1]
        cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)

        out.write(frame)

    out.release()


def ssd_match(template, image):
    return np.sum((template - image) ** 2)


def cc_match(template, image):
    return np.sum(template * image)


def ncc_match(template, image):
    template_mean = np.mean(template)
    image_mean = np.mean(image)
    template_std = np.std(template)
    image_std = np.std(image)
    return np.sum(((template - template_mean) * (image - image_mean)) / (template_std * image_std))


def main():
    base_path = 'video/image_girl/' 
    output_video_path = 'NCC.mp4'  
    total_frames = 500  # Total number of frames in the video

    # Format the path pattern with the base path and image file names
    video_path_pattern = base_path + '{:04d}.jpg' 
    
    # Format the initial frame path
    initial_frame_path = video_path_pattern.format(1)
    bbox = select_bounding_box(initial_frame_path)
    method = 'NCC'  # Choose 'SSD', 'CC', or 'NCC' as needed
    tracking_results = track_object(video_path_pattern, bbox, total_frames, method=method)
    create_video(video_path_pattern, output_video_path, tracking_results, total_frames)

if __name__ == "__main__":
    main()