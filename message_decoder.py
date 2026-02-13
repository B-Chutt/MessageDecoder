import tkinter as tk
from tkinter import filedialog
from tkinter.ttk import Progressbar
import math
import cv2
from collections import defaultdict
from PIL import Image, ImageTk, ImageDraw, ImageFont


class MessageDecoderGui:
    PATTERN_STABLE_STATE = 0
    CHANGE_IN_PROGRESS_STATE = 1
    DIAGRAM_SCALE = 3.5
    OUTPUT_WIDTH = 160
    OUTPUT_HEIGHT = 12

    def __init__(self, root):
        self.root = root
        self.root.title("Message Decoder")

        self.video = None
        self.paused = True

        # --- Layout frame for video + bitmap ---
        display_frame = tk.Frame(root, bd=3, relief=tk.RIDGE)
        display_frame.grid(row=0, column=0, padx=10, pady=10)

        # Video output
        self.video_label = tk.Label(display_frame)

        # --- Create internal bitmap ---
        self.diagram = Image.new("RGB", (320, 640), "white")
        self.bitmap_tk = ImageTk.PhotoImage(self.diagram)
        self.bitmap_label = tk.Label(display_frame, image=self.bitmap_tk)

        controls = tk.Frame(display_frame, bd=3, relief=tk.RIDGE)
        controls.grid(row=0, column=0)

        self.open_file_button = tk.Button(controls, text="Open MP4", command=self.open_file)
        self.map_led_button = tk.Button(controls, text="Map LEDs", command=self.get_led_positions)
        self.read_mess_button = tk.Button(controls, text="Read Message", command=self.get_characters)
        self.stop_button = tk.Button(controls, text="Stop", command=self.stop_operation)
        self.output_text = tk.Text(display_frame, width=self.OUTPUT_WIDTH, height=self.OUTPUT_HEIGHT)
        job_prog_var = tk.StringVar()
        job_prog_var.set("Job Progress")
        progress_frame = tk.Frame(controls, padx=10)
        progress_frame.grid(row=0, column=1, rowspan=5)
        job_prog_label = tk.Label(progress_frame, textvariable=job_prog_var)
        self.job_progress = Progressbar(progress_frame, length=200, mode='determinate')
        spacer_label = tk.Label(progress_frame)
        spacer_label.grid(row=2, column=0)
        video_prog_var = tk.StringVar()
        video_prog_var.set("Video File Progress")
        video_prog_label = tk.Label(progress_frame, textvariable=video_prog_var)
        self.video_progress = Progressbar(progress_frame, length=200, mode='determinate')
        self.video_label.grid(row=0, column=1, pady=10)
        self.bitmap_label.grid(row=0, column=2)
        self.open_file_button.grid(row=0, column=0, pady=10)
        self.map_led_button.grid(row=1, column=0, pady=10)
        self.read_mess_button.grid(row=2, column=0, padx=20, pady=10)
        self.stop_button.grid(row=3, column=0, pady=10)
        job_prog_label.grid(row=0, column=0)
        self.job_progress.grid(row=1, column=0)
        video_prog_label.grid(row=3, column=0)
        self.video_progress.grid(row=4, column=0)
        self.output_text.grid(row=1, column=0, columnspan = 3, padx=10, pady=10)

        self.map_led_button["state"] = "disabled"
        self.read_mess_button["state"] = "disabled"
        self.stop_button["state"] = "disabled"
        self.clear_output_text()
        self.led_positions = []
        self.sorted_sides = []
        self.message_data = []
        self.diag_centroid_x = 0
        self.diag_centroid_y = 0
        self.frame_number = 0
        self.total_frames = 0
        self.stable_count = 0
        self.char_extract_state = self.PATTERN_STABLE_STATE
        self.last_stable_frame_data = []
        self.last_frame_data = []
        self.font = ImageFont.truetype("arial.ttf", 30)
        self.out_cursor_x = 0
        self.out_cursor_y = 1
        self.message_frame = 0
        self.drop_everything = False

    def clear_output_text(self):
        self.output_text.delete("1.0", "end")
        # Fill the output widget with spaces to allow text to pe placed anywhere
        for linePos in range(1, self.OUTPUT_HEIGHT):
            for charpos in range(self.OUTPUT_WIDTH-1):
                self.output_text.insert("end", " ")
            self.output_text.insert("end", "\r\n")

    def stop_operation(self):
        self.drop_everything = True

    def set_out_cursor(self, line, column):
        self.out_cursor_x = column
        self.out_cursor_y = line

    def char_at_cursor(self,  character):
        self.output_text.delete(f"{self.out_cursor_y}.{self.out_cursor_x}", f"{self.out_cursor_y}.{self.out_cursor_x+1}")
        self.output_text.insert(f"{self.out_cursor_y}.{self.out_cursor_x}", character)
        self.out_cursor_x += 1

    def open_file(self):
        success = False
        self.clear_diagram()
        path = filedialog.askopenfilename(
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )
        if path:
            if self.video:
                # Shut down any already open videos
                self.video.release()
            self.video = cv2.VideoCapture(path)
            if self.video:
                ret, frame = self.video.read()
                if ret:
                    self.show_frame(frame)
                    self.map_led_button["state"] = "normal"
                    self.read_mess_button["state"] = "disabled"
                    success = True
        if not success:
            self.map_led_button["state"] = "disabled"
            self.read_mess_button["state"] = "disabled"
            self.video_label.configure(image="")
        return

    def update_frame(self):
        if self.paused or not self.video:
            return

        ret, frame = self.video.read()
        if not ret:
            return

        self.show_frame(frame)
        self.root.after(30, self.update_frame)

    def show_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(frame, (320, 640), interpolation=cv2.INTER_AREA)
        img = Image.fromarray(resized_frame)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def get_led_pos_one_frame(self):
        still_running, frame = self.video.read()
        if still_running:
            found_positions = self.get_lit_led_positions(frame)
            print("Frame " + str(self.frame_number) + ". " + str(len(found_positions)) + " lit LEDs found")
            new_led_count = 0
            for found_pos in found_positions:
                already_seen = 0
                for existing_pos in self.led_positions:
                    if (abs(found_pos[0] - existing_pos[0]) < 10) and (abs(found_pos[1] - existing_pos[1]) < 10):
                        already_seen += 1
                        if already_seen > 1:
                            print("ERROR LED stored multiple times")
                if already_seen == 0:
                    self.led_positions.append(found_pos)

                    # Show the LED on the diagram
                    ImageDraw.Draw(self.diagram).circle((found_pos[0]/self.DIAGRAM_SCALE, found_pos[1]/self.DIAGRAM_SCALE), 7, outline="lightgrey", width=4)
                    diagram_tk = ImageTk.PhotoImage(self.diagram)
                    self.bitmap_label.imgtk = diagram_tk
                    self.bitmap_label.configure(image=diagram_tk)

                    new_led_count += 1
            if new_led_count != 0:
                print("Found " + str(new_led_count) + " new LEDs")
            self.frame_number += 1
            self.show_frame(frame)
            leds_found = len(self.led_positions)
            self.job_progress["value"] = 100*leds_found/48
            self.video_progress["value"] = 100*self.frame_number/self.total_frames
            if leds_found >= 48:
                # We've got all the LED positions we're looking for. Hope they're all real!
                still_running = False

        if not still_running:
            # We've either run out of video, or we've found 48 LEDs
            self.split_and_sort_hexagon_sides()
            self.show_led_diagram()
            self.read_mess_button["state"] = "normal"
            self.stop_button["state"] = "disabled"
        else:
            if self.drop_everything:
                self.stop_button["state"] = "disabled"
            else:
                # Not finished yet. Give the GUI some time and then go again
                self.root.after(5, self.get_led_pos_one_frame)

    def get_lit_led_positions(self, frame):
        positions = []

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Threshold to find bright regions
        _, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

        # Find contours of bright regions
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw circles around bright areas
        for cnt in contours:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            if (radius > 10) and (y < 1420):
                positions.append((x, y))
        return positions

    def get_led_positions(self):
        self.led_positions = []

        if not self.video:
            print("Video file not open.")
            return
        self.total_frames = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_rate = self.video.get(cv2.CAP_PROP_FPS)
        width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print("Video file has " + str(self.total_frames) + " frames and runs at " + str(frame_rate)+" FPS")
        print("Frame size is " + str(width) + " X " + str(height)+".")
        self.clear_diagram()
        self.frame_number = 0
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.stop_button["state"] = "normal"
        self.read_mess_button["state"] = "disabled"
        self.drop_everything = False
        self.job_progress["value"]=0
        self.video_progress["value"]=0
        self.get_led_pos_one_frame()

    def clear_diagram(self):
        ImageDraw.Draw(self.diagram).rectangle((0, 0, 319, 639), "white")
        diagram_tk = ImageTk.PhotoImage(self.diagram)
        self.bitmap_label.imgtk = diagram_tk
        self.bitmap_label.configure(image=diagram_tk)

    def split_and_sort_hexagon_sides(self):
        """
        Given a list of (x, y) points that lie roughly on the sides of a hexagon,
        return six lists of points, each sorted anti-clockwise along the hexagon.
        """

        # --- 1. Compute centroid ---
        centroid_x = sum(p[0] for p in self.led_positions) / len(self.led_positions)
        centroid_y = sum(p[1] for p in self.led_positions) / len(self.led_positions)

        # --- 2. Compute angle of each point relative to centroid ---
        angle_data = []
        for p in self.led_positions:
            dx = p[0] - centroid_x
            dy = p[1] - centroid_y
            ang = math.atan2(dy, dx)
            ang += (math.pi/6) # Offset by half a side
            ang = (ang + 2*math.pi) % (2*math.pi)
            angle_data.append((p, ang))

        # --- 3. Assign to 6 angular sectors ---
        sector_size = 2 * math.pi / 6
        sides = defaultdict(list)
        for p, ang in angle_data:
            sector = int(ang // sector_size)
            sides[sector].append((p, ang))

        # --- 4. Sort the sides themselves anti-clockwise ---
        self.sorted_sides = []
        for i in range(6):
            side = sides[i]

            # Sort points within each side by angle (anti-clockwise)
            side_sorted = sorted(side, key=lambda x: x[1])

            # Keep only the points, not the angles
            self.sorted_sides.append([p for p, _ in side_sorted])

        # Store the center of the hexagon on the diagram bitmap
        self.diag_centroid_x = centroid_x / self.DIAGRAM_SCALE
        self.diag_centroid_y = centroid_y / self.DIAGRAM_SCALE

    def show_led_diagram(self, current_state=None):
        if current_state is None:
            current_state = [0xff, 0xff, 0xff, 0xff, 0xff, 0xff]
        side_colours = ["red", "magenta", "green", "orange", "yellow", "blue"]
        draw = ImageDraw.Draw(self.diagram)
        side_index = 0
        for side in self.sorted_sides:
            bit_weight = 1
            for ledPos in side:
                side_colour = side_colours[side_index]
                if current_state[side_index] & bit_weight:
                    # LED on
                    draw.circle((ledPos[0]/self.DIAGRAM_SCALE, ledPos[1]/self.DIAGRAM_SCALE), 7, side_colour, "lightgrey", width=2)
                else:
                    # LED off
                    draw.circle((ledPos[0] / self.DIAGRAM_SCALE, ledPos[1] / self.DIAGRAM_SCALE), 7, "lightgrey",
                                "lightgrey", width=2)

                bit_weight <<= 1
            side_index += 1
        diagram_tk = ImageTk.PhotoImage(self.diagram)
        self.bitmap_label.imgtk = diagram_tk
        self.bitmap_label.configure(image=diagram_tk)

    def get_characters(self):
        self.message_data = []
        if not self.video:
            print("Video file not open.")
            return

        self.frame_number = 0
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.stable_count = 0
        self.char_extract_state = self.PATTERN_STABLE_STATE
        self.last_stable_frame_data = []
        self.clear_output_text()
        self.message_frame = 0

        # Look for pattern changes. Need to account for the fact that a pattern change normally results in at-least one
        # frame of rubbish.
        # To be taken as a new pattern, we need to see a change from one
        # frame to the next, followed by a number of identical frames in a row
        got_frame, frame = self.video.read()
        if got_frame:
            self.last_stable_frame_data = self.get_frame_data(frame)
            self.message_data.append(self.last_stable_frame_data)
            self.drop_everything = False
            self.stop_button["state"] = "normal"
            self.job_progress["value"]=0
            self.video_progress["value"]=0
            self.get_characters_one_frame()

    def get_characters_one_frame(self):
        got_frame, frame = self.video.read()
        if got_frame:
            draw = ImageDraw.Draw(self.diagram)
            new_frame_data = self.get_frame_data(frame)
            if self.char_extract_state == self.PATTERN_STABLE_STATE:
                if new_frame_data != self.last_stable_frame_data:
                    self.stable_count = 0
                    self.last_frame_data = new_frame_data
                    self.char_extract_state = self.CHANGE_IN_PROGRESS_STATE
            elif self.char_extract_state == self.CHANGE_IN_PROGRESS_STATE:
                if new_frame_data == self.last_frame_data:
                    self.stable_count += 1
                    if self.stable_count > 3:
                        self.show_frame(frame)
                        self.show_message_rotations(new_frame_data)
                        ImageDraw.Draw(self.diagram).rectangle((0, 0, 319, 639), "white")
                        self.show_led_diagram(new_frame_data)
                        for side_index in range(6):
                            current_side_pos = self.sorted_sides[side_index]
                            character = chr(new_frame_data[side_index])
                            char_x = ((current_side_pos[3][0] / self.DIAGRAM_SCALE)+self.diag_centroid_x)/2
                            char_y = ((current_side_pos[3][1] / self.DIAGRAM_SCALE)+self.diag_centroid_y)/2
                            draw.text((char_x, char_y), character, fill="black", font=self.font, anchor="mm")
                            side_index += 1
                        diagram_tk = ImageTk.PhotoImage(self.diagram)
                        self.bitmap_label.imgtk = diagram_tk
                        self.bitmap_label.configure(image=diagram_tk)
                        self.message_data.append(self.last_frame_data)
                        self.last_stable_frame_data = self.last_frame_data
                        self.char_extract_state = self.PATTERN_STABLE_STATE
                else:
                    self.stable_count = 0
                    self.last_frame_data = new_frame_data
            if self.drop_everything:
                # Stop button pressed
                self.stop_button["state"] = "disabled"
            else:
                self.frame_number += 1
                self.job_progress["value"] = 100*self.frame_number/self.total_frames
                self.video_progress["value"] = 100*self.frame_number/self.total_frames
                self.root.after(10, self.get_characters_one_frame)
        else:
            # End of file
            self.stop_button["state"] = "disabled"

    def show_message_rotations(self, one_frame_chars):
        for rotation in range(6):
            side_index = rotation
            self.set_out_cursor(1 + rotation, 6 * self.message_frame)
            for char_num in range(6):
                self.char_at_cursor(chr(one_frame_chars[side_index]))
                side_index += 1
                if side_index >= 6:
                    side_index = 0
        self.message_frame += 1

    def get_frame_data(self, frame):
        # Convert to grayscale
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        output_bytes = []
        for side in self.sorted_sides:
            value = 0
            bit_weight = 1
            for led in side:
                if grey[int(led[1]), int(led[0])] > 220:
                    value += bit_weight
                bit_weight *= 2
            output_bytes.append(value)
        return output_bytes


_root = tk.Tk()
MessageDecoderGui(_root)
_root.mainloop()
