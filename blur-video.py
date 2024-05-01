# This code randomly selects two 1-second clips from the video and blurs them
# in order to test whether the uncertainty score really estimates the uncertainty
# If it does, the blurred frames should have higher uncertainty scores
from skimage.filters import gaussian
from moviepy.editor import VideoFileClip, concatenate_videoclips
import random
import os


def blur_video(video_name):
    clip = VideoFileClip(os.path.join(VIDEO_DIR, video_name))
    duration = clip.duration
    start_time1 = 0
    start_time2 = 0
    while True:
        start_time1 = random.uniform(0, duration - 1)
        start_time2 = random.uniform(0, duration - 1)
        if start_time1 > start_time2:
            start_time1, start_time2 = start_time2, start_time1
        if start_time2 - start_time1 > 1:
            break

    print(start_time1, start_time2, duration)

    before_blur = clip.subclip(0, start_time1)
    blur_segment1 = clip.subclip(start_time1, start_time1 + 1).fl_image(lambda img: gaussian(img.astype(float), sigma=(11, 11), channel_axis=2))
    between_blur = clip.subclip(start_time1 + 1, start_time2)
    blur_segment2 = clip.subclip(start_time2, start_time2 + 1).fl_image(lambda img: gaussian(img.astype(float), sigma=(11, 11), channel_axis=2))
    after_blur = clip.subclip(start_time2 + 1, duration)

    final_clip = concatenate_videoclips([before_blur, blur_segment1, between_blur, blur_segment2, after_blur])
    print(os.path.join(OUTPUT_VIDEO_DIR, video_name))
    final_clip.write_videofile(os.path.join(OUTPUT_VIDEO_DIR, video_name))

    print(f"{video_name} has been blurred.")


VIDEO_DIR = "./video_bipeng"
OUTPUT_VIDEO_DIR = "./video_blurred"
blur_video("53-30-360x480.mp4")
