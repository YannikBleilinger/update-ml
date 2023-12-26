import websocket
import pytest
import time

@pytest.fixture()
def client():
    with websocket.app.test_client() as client:
        yield client


def test_scan(client):
    image_path = "test\\testResources\\img1.jpg"
    
    with open(image_path, "rb") as image:
        start_time = time.time()
        response = client.post('/scanImage', data={'image': (image, 'test_image.jpg')})
        print("--- %s seconds ---" % (time.time() - start_time))

    print(response.status_code)


    assert response.status_code == 200

