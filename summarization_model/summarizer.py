import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def summarize_session(transcript: str) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # If transcript is short enough, summarize directly
    if len(transcript) < 12000:
        return _call_groq(client, transcript)

    # Otherwise: split → summarize each chunk → combine summaries
    print("📄 Long transcript detected — summarizing in chunks...")
    chunk_size = 10000
    chunks = [transcript[i:i+chunk_size] for i in range(0, len(transcript), chunk_size)]

    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"📝 Summarizing chunk {i+1}/{len(chunks)}...")
        summary = _call_groq(client, chunk, is_chunk=True)
        chunk_summaries.append(summary)

    # Final pass: summarize all chunk summaries into one report
    print("🔗 Combining all chunks into final report...")
    combined = "\n\n".join(chunk_summaries)
    return _call_groq(client, combined, is_final=True)


def _call_groq(client, text: str, is_chunk=False, is_final=False) -> str:
    if is_chunk:
        instruction = """
أنت مساعد أكاديمي متخصص في تلخيص المحاضرات.
لخص هذا الجزء من المحاضرة بالشكل التالي:
- اذكر كل نقطة أو مفهوم تم شرحه
- اشرح كل نقطة في جملة أو جملتين توضح المقصود منها
- لا تكتف بذكر اسم المفهوم فقط، بل وضح ما قاله الدكتور عنه
- تجنب التكرار
"""
    elif is_final:
        instruction = """
أنت مساعد أكاديمي. لديك ملخصات أجزاء من محاضرة واحدة.
اجمعها في تقرير نهائي منظم بالتنسيق التالي بالضبط:

## 📚 موضوع المحاضرة
[اكتب موضوع المحاضرة الرئيسي في جملة واحدة]

## 🗂️ المواضيع التي تم تناولها
[قائمة بالمواضيع الرئيسية فقط]

## 📝 شرح النقاط المناقشة
لكل نقطة أو مفهوم تم شرحه في المحاضرة، اكتبها بهذا الشكل:

**[اسم المفهوم]**
← [اشرح ما قاله الدكتور عن هذا المفهوم في 2-3 جمل. وضح الفكرة الأساسية وكيف تعمل أو تُستخدم]

## ⚠️ نقاط مهمة نبّه عليها الدكتور
[أي تحذيرات أو نقاط أكد عليها الدكتور بشكل خاص]

## ✅ خلاصة
[فقرة قصيرة تلخص المحاضرة كاملة في 3-4 جمل]

قواعد مهمة:
- لا تكتب "تمت مناقشة" قبل كل نقطة
- لا تكرر نفس المعلومة في أكثر من مكان
- اشرح كل مفهوم بشكل مفيد , بالتفصيل وليس مجرد ذكر اسمه
- اكتب بالعربية الفصحى البسيطة
- إذا كان المصطلح أو المفهوم إنجليزياً في الأصل، اكتبه بالإنجليزية وليس بالعربية (مثال: اكتب "Round Function" وليس "الراوند فانكشن"، اكتب "XOR" وليس "الاكسور"، اكتب "Plaintext" وليس "البلينتكست")
- اذا كان الموضوع غير واضح لا تفترض من عندك
"""
    else:
        instruction = """
أنت مساعد أكاديمي متخصص في تلخيص المحاضرات الجامعية.
لديك نص محاضرة جامعية باللغة العربية المصرية.

اكتب تقريراً منظماً بالتنسيق التالي بالضبط:

## 📚 موضوع المحاضرة
[اكتب موضوع المحاضرة الرئيسي في جملة واحدة]

## 🗂️ المواضيع التي تم تناولها
[قائمة بالمواضيع الرئيسية فقط]

## 📝 شرح النقاط المناقشة
لكل نقطة أو مفهوم تم شرحه في المحاضرة، اكتبها بهذا الشكل:

**[اسم المفهوم]**
← [اشرح ما قاله الدكتور عن هذا المفهوم في 2-3 جمل. وضح الفكرة الأساسية وكيف تعمل أو تُستخدم]

## ⚠️ نقاط مهمة نبّه عليها الدكتور
[أي تحذيرات أو نقاط أكد عليها الدكتور بشكل خاص]

## ✅ خلاصة
[فقرة قصيرة تلخص المحاضرة كاملة في 3-4 جمل]

قواعد مهمة:
- لا تكتب "تمت مناقشة" قبل كل نقطة
- لا تكرر نفس المعلومة في أكثر من مكان  
- اشرح كل مفهوم بشكل مفيد وليس مجرد ذكر اسمه
- اكتب بالعربية الفصحى البسيطة
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "أنت مساعد أكاديمي محترف متخصص في تلخيص المحاضرات الجامعية. تشرح المفاهيم بوضوح وإيجاز دون تكرار."
            },
            {
                "role": "user",
                "content": f"{instruction}\n\nنص المحاضرة:\n{text}"
            }
        ],
        temperature=0.2,  # lower = more focused, less hallucination
        max_tokens=3000   # increased to allow detailed explanations
    )
    return response.choices[0].message.content