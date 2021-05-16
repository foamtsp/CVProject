from fastapi import FastAPI, File, UploadFile, status, HTTPException
from model.generator import FontGenerator
import requests
import pathlib
import cv2

# service entity
app = FastAPI()
font_generator = FontGenerator()


@app.post('/genfont/{img_id}', status_code=status.HTTP_201_CREATED)
async def generate_font_from_sample_image(img_id: str, file: UploadFile = File(...)):

    # get image
    file.filename = f"{img_id}_sample.jpg"
    contents = await file.read()  # <-- Important!
    pathlib.Path("buffer").mkdir(parents=True, exist_ok=True)
    path = pathlib.Path("buffer") / file.filename
    # example of how you can save the file
    with open(path, "wb") as f:
        f.write(contents)

    # read image as numpy array
    sample_chars = cv2.imread(path, 0)

    # gen font
    all_char_panel = font_generator.generate_font_from_sample_image(sample_chars)
    output_path = pathlib.Path("buffer") / f"{img_id}_font.jpg"
    cv2.imwrite(output_path, all_char_panel)

    # send to save in cloud storage
    if all_char_panel:
        # send as a form data (Top task)
        r = requests.post(f"dummy_url/image/result/{img_id}", files="xxxxx")
        if r.status_code == 200:
            return {
                "status": "success"
            }
        else:
            raise HTTPException(status_code=404, detail="Resource Not found")




