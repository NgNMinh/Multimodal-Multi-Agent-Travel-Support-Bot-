from typing import Literal
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import HumanMessage, ToolMessage
from src.core.nodes import graph
import base64
from groq import Groq
import os 
import chainlit as cl
from io import BytesIO
import numpy as np 
import io
import audioop
import wave


# give me tickets from hanoi to saigon on 30 april 2025
@cl.on_message
async def on_message(msg: cl.Message):
    config = {"configurable": {"user_id": "67dac00bd70c58a5d543e4c2", "thread_id": cl.context.session.id}}
    cb = cl.LangchainCallbackHandler()
    final_answer = cl.Message(content="")
    
    # Process any attached images
    content = msg.content
    if msg.elements:
        with open(msg.elements[0].path, "rb") as f:
            image_bytes = f.read()
            # Convert image to base64
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": content},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ]
            client = Groq(
                api_key=os.getenv("GROQ_API_KEY"),
                )
            response = client.chat.completions.create(
                model="llama-3.2-90b-vision-preview",
                messages=messages,
                max_tokens=1000,
            )
            content += f"\n[Image Analysis: {response.choices[0].message.content}]"
            print(content)
    
    for msg, metadata in graph.stream({"messages": content}, stream_mode="messages",
                                      config=RunnableConfig(callbacks=[cb], **config)):
        if (
                msg.content
                and not isinstance(msg, HumanMessage)
                and  not isinstance(msg, ToolMessage)
        ):
            await final_answer.stream_token(msg.content)

    await final_answer.send()


@cl.on_audio_start
async def on_audio_start():
    cl.user_session.set("audio_chunks", [])
    
    return True

@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    audio_chunks = cl.user_session.get("audio_chunks")
    if audio_chunks is not None:
        audio_chunk = np.frombuffer(chunk.data, dtype=np.int16)
        audio_chunks.append(audio_chunk)

@cl.on_audio_end
async def on_audio_end():
    audio_chunks = cl.user_session.get("audio_chunks")
    if not audio_chunks:
        return

    # Gộp các chunk lại
    concatenated = np.concatenate(audio_chunks)
    wav_buffer = io.BytesIO()

    # Ghi thành file WAV
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(concatenated.tobytes())

    wav_buffer.seek(0)
    audio_buffer = wav_buffer.getvalue()
    cl.user_session.set("audio_chunks", [])

    # Tạo audio element để hiển thị
    input_audio_el = cl.Audio(content=audio_buffer, mime="audio/wav")

    # Gửi đi để transcribe
    whisper_input = ("audio.wav", audio_buffer, "audio/wav")

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    transcription = client.audio.transcriptions.create(
        file=whisper_input,
        model="whisper-large-v3-turbo",
        language='en',
        response_format="text",
    )
    print(transcription)
    await cl.Message(
        author="You",
        type="user_message",
        content=transcription,
        elements=[input_audio_el],
    ).send()

    # Tạo message mới với transcript
    new_msg = cl.Message(content=transcription)

    # Gửi transcript đến chatbot
    await on_message(new_msg)