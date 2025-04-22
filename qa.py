import openai

openai.api_key = "your_openai_api_key"  # Replace this

def get_answer(question, chunks, index, embeddings, embedder, k=5):
    question_vec = embedder(question)
    _, indices = index.search(question_vec, k)
    context = ' '.join([chunks[i] for i in indices[0]])

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Answer the user's question using only the provided context."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ]
    )

    return response['choices'][0]['message']['content']
