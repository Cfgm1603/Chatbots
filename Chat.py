import os
import warnings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

warnings.filterwarnings("ignore")

# 🔐 Cargar variables desde el archivo .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("❌ No se encontró OPENAI_API_KEY en el archivo .env")

# 🤖 Configuración del modelo OpenRouter
llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model_name="mistralai/mistral-7b-instruct",
    temperature=0.7,
)

Meta_prompt = "Actúa como un coach de alto rendimiento para correr, eres maratonista"
conversation_history = []
conversation_summary = ""
MAX_MESSAGES = 3 

def generar_resumen(historial):
    """Genera un resumen conciso de la conversación usando la LLM"""
    if not historial:
        return ""
    
    conversacion_texto = "\n".join([
        f"{'Usuario' if i % 2 == 0 else 'Asistente'}: {msg}"
        for i, msg in enumerate(historial)
    ])
    
    prompt_resumen = f"""Resume la siguiente conversación de manera concisa, capturando los puntos clave, 
    el contexto importante y cualquier información relevante para continuar la conversación. 
    Mantén el resumen en máximo 3-4 oraciones.

Conversación:
{conversacion_texto}

Resumen conciso:"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt_resumen)])
        return response.content.strip()
    except Exception as e:
        print(f"⚠️ Error al generar resumen: {e}")
        return ""

# 🗨️ Bucle de chat con memoria
print("💬 Chatbot Mistral vía OpenRouter (escribe 'salir' para terminar)")
print("📊 Sistema de resumen automático activado\n")

while True:
    user_input = input("👤 Tú: ")
    
    if user_input.lower() in ["salir", "exit", "quit"]:
        print("👋 Hasta luego.")
        break

    # Guardar mensaje del usuario
    conversation_history.append(user_input)
    
    # Construir el contexto completo
    if conversation_summary:
        contexto = f"{Meta_prompt}\n\nResumen de conversación previa: {conversation_summary}\n\nContinúa la conversación: {user_input}"
    else:
        contexto = f"{Meta_prompt}\n\n{user_input}"
    
    try:
        
        response = llm.invoke([HumanMessage(content=contexto)])
        bot_response = response.content.strip()
        
        
        conversation_history.append(bot_response)
        
        print(f"🤖 Bot: {bot_response}\n")
        
        
        if len(conversation_history) >= MAX_MESSAGES:
            print("📝 Generando resumen de la conversación...")
            conversation_summary = generar_resumen(conversation_history)
            conversation_history = []  # Limpiar historial después del resumen
            print(f"✅ Resumen generado: {conversation_summary[:100]}...\n")

    except Exception as e:
        print(f"❌ Error: {e}\n")