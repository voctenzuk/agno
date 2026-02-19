from types import SimpleNamespace

from agno.models.openai.chat import OpenAIChat


def _fake_chat_completion(content, model_extra=None):
    message = SimpleNamespace(
        role="assistant",
        content=content,
        tool_calls=None,
        audio=None,
        model_extra=model_extra,
    )
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(
        choices=[choice],
        usage=None,
        id="resp_1",
        system_fingerprint=None,
        model_extra=None,
        error=None,
    )


def test_parse_provider_response_extracts_images_from_multipart_content():
    model = OpenAIChat(id="gpt-4o-mini")

    response = _fake_chat_completion(
        content=[
            {"type": "text", "text": "Вот ваше изображение"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,aGVsbG8="},
            },
        ],
    )

    model_response = model._parse_provider_response(response)  # type: ignore[arg-type]

    assert model_response.content == "Вот ваше изображение"
    assert model_response.images is not None
    assert len(model_response.images) == 1
    assert model_response.images[0].mime_type == "image/png"
    assert model_response.images[0].content == b"hello"


def test_parse_provider_response_multipart_merges_with_model_extra_images():
    model = OpenAIChat(id="gpt-4o-mini")

    response = _fake_chat_completion(
        content=[
            {"type": "text", "text": "two images"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,aGVsbG8="},
            },
        ],
        model_extra={
            "images": [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.png"},
                }
            ]
        },
    )

    model_response = model._parse_provider_response(response)  # type: ignore[arg-type]

    assert model_response.content == "two images"
    assert model_response.images is not None
    assert len(model_response.images) == 2
    assert model_response.images[0].content == b"hello"
    assert model_response.images[1].url == "https://example.com/image.png"


def test_parse_provider_response_string_content_unchanged():
    model = OpenAIChat(id="gpt-4o-mini")

    response = _fake_chat_completion(content="plain text")

    model_response = model._parse_provider_response(response)  # type: ignore[arg-type]

    assert model_response.content == "plain text"
    assert model_response.images is None


def test_parse_provider_response_extracts_output_text_and_string_image_url():
    model = OpenAIChat(id="gpt-4o-mini")

    response = _fake_chat_completion(
        content=[
            {"type": "output_text", "text": "image from url"},
            {"type": "image_url", "image_url": "https://example.com/generated.png"},
        ],
    )

    model_response = model._parse_provider_response(response)  # type: ignore[arg-type]

    assert model_response.content == "image from url"
    assert model_response.images is not None
    assert len(model_response.images) == 1
    assert model_response.images[0].url == "https://example.com/generated.png"


def test_parse_provider_response_ignores_invalid_data_uri_image():
    model = OpenAIChat(id="gpt-4o-mini")

    response = _fake_chat_completion(
        content=[
            {"type": "text", "text": "bad image"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,%%%"}},
        ],
    )

    model_response = model._parse_provider_response(response)  # type: ignore[arg-type]

    assert model_response.content == "bad image"
    assert model_response.images is None


def test_parse_provider_response_model_extra_only_still_works():
    model = OpenAIChat(id="gpt-4o-mini")

    response = _fake_chat_completion(
        content="legacy path",
        model_extra={
            "images": [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/legacy.png"},
                }
            ]
        },
    )

    model_response = model._parse_provider_response(response)  # type: ignore[arg-type]

    assert model_response.content == "legacy path"
    assert model_response.images is not None
    assert len(model_response.images) == 1
    assert model_response.images[0].url == "https://example.com/legacy.png"
