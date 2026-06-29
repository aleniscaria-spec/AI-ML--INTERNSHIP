import face_recognition as fr
from PIL import Image, ImageDraw

known_students = {
    "student1": "Attendance/dohni.jpg",
    "student2": "Attendance/virat.jpg"}


known_encodings = {}
for name, file in known_students.items():
    image = fr.load_image_file(file)
    encodings = fr.face_encodings(image)
    if len(encodings) > 0:
        known_encodings[name] = encodings[0]
    else:
        print(f"No face found in {file}")

group_image = fr.load_image_file("Attendance/group.png")

face_locations = fr.face_locations(group_image)
print("Total faces detected:", len(face_locations))
if len(face_locations) == 0:
    print("No students detected")
    exit()

group_encodings = fr.face_encodings(group_image, face_locations)
pil_image = Image.fromarray(group_image)
draw = ImageDraw.Draw(pil_image)
attendance = {}
present_count = 0

for student_name, known_encoding in known_encodings.items():
    found = False
    for location, encoding in zip(face_locations, group_encodings):
        match = fr.compare_faces([known_encoding], encoding)
        if match[0]:
            found = True
            present_count += 1
            top, right, bottom, left = location

            draw.rectangle(
                [(left, top), (right, bottom)],outline="green", width=5)
            draw.text((left, top - 20), student_name, fill="green")
            break
    attendance[student_name] = found

print("*" * 50)
print("Attendance Report")
for student, status in attendance.items():
    if status:
        print(f"{student} - Present")
    else:
        print(f"{student} - Absent")

print(f"\nPresent: {present_count}/{len(attendance)}")
print("*" * 50)

pil_image.save("Attendance/attendance.png")
print("attendance.png saved successfully.")