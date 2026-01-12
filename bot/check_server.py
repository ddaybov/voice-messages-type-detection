"""
Скрипт для проверки доступности сервера.
"""

import os
import sys
import asyncio
import aiohttp

SERVER_URL = os.getenv("SERVER_URL", "http://127.0.0.1:8000")
API_URL = f"{SERVER_URL.rstrip('/')}/predict" if not SERVER_URL.endswith("/predict") else SERVER_URL
HEALTH_URL = f"{SERVER_URL.rstrip('/').replace('/predict', '')}/health"
SUPPORTED_FORMATS_URL = f"{SERVER_URL.rstrip('/').replace('/predict', '')}/supported_formats"


async def check_health():
    """Проверить /health endpoint."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(HEALTH_URL, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Сервер доступен: {HEALTH_URL}")
                    print(f"   Статус: {data.get('status', 'unknown')}")
                    return True
                else:
                    print(f"❌ Сервер вернул статус {response.status}")
                    return False
    except aiohttp.ClientError as e:
        print(f"❌ Не удалось подключиться к серверу: {e}")
        print(f"   URL: {HEALTH_URL}")
        return False
    except Exception as e:
        print(f"❌ Ошибка при проверке здоровья сервера: {e}")
        return False


async def check_supported_formats():
    """Проверить /supported_formats endpoint."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(SUPPORTED_FORMATS_URL, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"\n✅ Поддерживаемые форматы:")
                    print(f"   Аудио: {', '.join(data.get('audio_formats', []))}")
                    print(f"   Языки: {', '.join(data.get('languages', {}).keys())}")
                    return True
                else:
                    print(f"❌ Не удалось получить поддерживаемые форматы (статус {response.status})")
                    return False
    except Exception as e:
        print(f"❌ Ошибка при проверке форматов: {e}")
        return False


async def check_predict_endpoint():
    """Проверить доступность /predict endpoint."""
    try:
        async with aiohttp.ClientSession() as session:
            # Просто проверяем, что endpoint существует (без отправки файла)
            # Отправляем пустой запрос, чтобы проверить доступность
            async with session.post(
                API_URL,
                data=aiohttp.FormData(),
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                # Ожидаем ошибку валидации (400/422), но это значит endpoint доступен
                if response.status in (400, 422, 413, 415):
                    print(f"✅ Endpoint /predict доступен (ожидаемая ошибка валидации)")
                    return True
                elif response.status == 200:
                    print(f"✅ Endpoint /predict доступен")
                    return True
                else:
                    print(f"⚠️  Endpoint /predict вернул неожиданный статус: {response.status}")
                    return False
    except aiohttp.ClientError as e:
        print(f"❌ Не удалось подключиться к /predict: {e}")
        print(f"   URL: {API_URL}")
        return False
    except Exception as e:
        print(f"❌ Ошибка при проверке /predict: {e}")
        return False


async def main():
    """Основная функция проверки."""
    print("🔍 Проверка доступности сервера...\n")
    print(f"Базовый URL: {SERVER_URL}")
    print(f"Health URL: {HEALTH_URL}")
    print(f"Predict URL: {API_URL}\n")
    
    results = []
    
    # Проверка здоровья
    print("1. Проверка /health...")
    results.append(await check_health())
    
    # Проверка поддерживаемых форматов
    print("\n2. Проверка /supported_formats...")
    results.append(await check_supported_formats())
    
    # Проверка predict endpoint
    print("\n3. Проверка /predict...")
    results.append(await check_predict_endpoint())
    
    # Итог
    print("\n" + "="*50)
    if all(results):
        print("✅ Все проверки пройдены! Сервер работает корректно.")
        sys.exit(0)
    else:
        print("❌ Некоторые проверки не прошли. Проверьте, что сервер запущен:")
        print("   uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
