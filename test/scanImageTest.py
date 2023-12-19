from websocket import app
import pytest
import io


@pytest.fixture()
def client():
    with app.test_client() as client:
        yield client


def test_scan(client):
    image_path = "test\testResources\img1.jpg"

    image_file = io.BytesIO(image_path)

    response = client.post('/scanImage', data={'image': (image_file, 'test_image.jpg')})

    print(response.statusCode)
