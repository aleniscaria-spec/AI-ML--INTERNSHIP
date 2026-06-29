import face_recognition as fr
from PIL import Image, ImageDraw


known_students = {
    "student1": "student1.png",
    "student2": "student2.jpg"}

known_encodings = {}
for name, file in known_students.items():
    image = fr.load_image_file(file)
    encodings = fr.face_encodings(image)

    if len(encodings) > 0:
        known_encodings[name] = encodings[0]
    else:
        print(f"No face found in {file}")


group_image = fr.load_image_file("class.jpg")
face_locations = fr.face_locations(group_image)
print("Total faces detected:", len(face_locations))

if len(face_locations) == 0:
    print("No students detected")
    exit()

group_encodings = fr.face_encodings(group_image, face_locations)

attendance = {}
present_locations = []
for student_name, known_encoding in known_encodings.items():

    found = False

    for location, encoding in zip(face_locations, group_encodings):

        match = fr.compare_faces([known_encoding], encoding)

        if match[0]:
            found = True
            present_locations.append(location)
            break
    attendance[student_name] = found

print("\nAttendance Report")
present_count = 0
for student, status in attendance.items():

    if status:
        print(f"{student} - Present")
        present_count += 1
    else:
        print(f"{student} - Absent")

print(f"\nPresent: {present_count}/{len(attendance)}")

pil_image = Image.fromarray(group_image)
draw = ImageDraw.Draw(pil_image)

for (top, right, bottom, left) in present_locations:
    draw.rectangle(
[(left, top), (right, bottom)],outline="green",width=3 )

del draw

pil_image.save("attendance.png")

print("\nattendance.png saved successfully.")