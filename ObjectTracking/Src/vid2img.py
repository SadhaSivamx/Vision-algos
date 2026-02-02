import cv2
import os


def extract(input_path, output_folder, n=10):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return

    frame_count = 0
    saved_count = 0

    print(f"Processing: {input_path} ...")

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % n == 0:
            height, width, _ = frame.shape
            frame = cv2.resize(frame, (width // 2, height // 2))
            filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1

            if saved_count % 50 == 0:
                print(f"Saved {saved_count} images so far...")

        frame_count += 1
    cap.release()
    print("--- process complete ---")
    print(f"Total frames scanned: {frame_count}")
    print(f"Total images saved: {saved_count}")
    print(f"Location: {output_folder}")


if __name__ == "__main__":

    extract('Drone.mp4', 'Frames/', n=1)
