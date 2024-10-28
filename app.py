from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import PyPDF2 as pdf
from dotenv import load_dotenv
import google.generativeai as genai
import numpy
import PyPDF2 as pdf

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-pro')


def get_input_prompt(extracted_text, jd):
    # Prompt Template
    input_prompt = f"""
    You are a skilled and very experienced ATS(Application Tracking System) with a deep understanding of tech field, software engineering,
    data science, data analyst, big data, and machine learning. Your task is to evaluate the resume based on the given job description.
    You must consider the job market is very competitive and you should provide best assistance for improving the resumes. 
    Assign the percentage Matching based on Job description and the missing keywords with high accuracy and anti-fraud feature where it can detect word spam in resume to match the vectorizer cosine similarity AI algorithm.

    Resume:{extracted_text}
    Description:{jd}

    I want the only response as follows in Indonesian Language and follow the format:

    "Berdasarkan hasil screening test untuk kandidat bernama [Nama Kandidat], berikut adalah hasil dan penjelasan yang berkaitan dengan kecocokan dan rekomendasi peningkatan kompetensi."
    •	Kecocokan: Berdasarkan analisis, keterampilan Anda dalam [sebutkan keterampilan yang cocok] sesuai dengan persyaratan yang diharapkan untuk posisi ini. Misalnya, kemampuan Anda dalam [contoh keterampilan] mencerminkan kecocokan yang kuat dengan peran ini. Hal ini menunjukkan bahwa Anda telah memiliki fondasi yang baik dalam [sebutkan bidang keterampilan], yang diperlukan untuk peran ini.
    •	Kata Kunci yang Hilang: Analisis menunjukkan bahwa ada beberapa keterampilan atau pengalaman yang belum disebutkan dalam CV Anda, yang mungkin relevan dengan posisi ini. Beberapa keterampilan yang hilang adalah [sebutkan kata kunci atau keterampilan]. Misalnya, kemampuan dalam [contoh keterampilan yang hilang] sangat penting untuk mendukung peran ini.
    •	Feedback untuk Peningkatan: Untuk meningkatkan peluang Anda, kami menyarankan agar Anda mengembangkan keterampilan dalam [sebutkan keterampilan yang hilang]. Contohnya, Anda dapat meningkatkan kompetensi dalam [contoh keterampilan] melalui kursus online, sertifikasi, atau pelatihan praktis. Mengembangkan keterampilan ini tidak hanya akan memperkuat posisi Anda dalam proses rekrutmen, tetapi juga meningkatkan kemampuan Anda dalam menavigasi tantangan yang terkait dengan peran ini.
    •	Hubungan dengan Psikologi Rekrutmen: Berdasarkan prinsip job-person fit dalam psikologi rekrutmen, keterampilan yang hilang ini bisa menunjukkan area pengembangan yang dapat membantu Anda lebih baik dalam menyelaraskan kemampuan kognitif dengan kebutuhan pekerjaan. Dengan meningkatkan keterampilan ini, Anda dapat memperkuat keselarasan antara profil Anda dan peran yang diharapkan oleh perusahaan.
    •	Contoh Tindakan yang Disarankan: Kami menyarankan agar Anda mengikuti kursus seperti [sebutkan kursus atau pelatihan terkait], atau menambah pengalaman praktis di bidang [sebutkan bidang]. Langkah-langkah ini akan membantu Anda lebih kompetitif dan memberikan kontribusi yang lebih signifikan dalam peran yang Anda lamar.

    """
    return input_prompt


app = Flask(__name__)
CORS(app)


@app.route("/")
def welcome():
    return "<p>Welcome To JustHire AI API</p>"


@app.route("/ATS", methods=["POST"])
def ATS():
    client_name = request.form["client_name"]
    job_desc = request.form["job_description"]

    fileList = request.files.getlist("resumeFiles[]")
    resume_list = []
    for i in fileList:
        print(i)
        filename = i.filename
        i.save('uploads/' + filename)

        reader = pdf.PdfReader(i)
        extracted_text = ""

        for page in range(len(reader.pages)):
            page = reader.pages[page]
            extracted_text += str(page.extract_text())

        text_array = [extracted_text, job_desc]

        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer()
        count_matrix = cv.fit_transform(text_array)

        from sklearn.metrics.pairwise import cosine_similarity
        match = cosine_similarity(count_matrix)[0][1]
        match *= 100
        match = round(match, 2)

        is_match = False
        if match < 70:
            is_match = False
        else:
            is_match = True

        input_prompt = get_input_prompt(extracted_text, job_desc)
        response = model.generate_content(input_prompt)

        resume_info = {
            "file_name": filename,
            "is_match": is_match,
            "job_match": match,
            "extracted_text": extracted_text,
            "response_from_ATS": response.text
        }

        resume_list.append(resume_info)

    return_data = {
        "client_name": client_name,
        "job_description": job_desc,
        "resumeProcessed": resume_list
    }

    return jsonify(return_data)


if __name__ == '__main__':
    app.run(debug=True)
