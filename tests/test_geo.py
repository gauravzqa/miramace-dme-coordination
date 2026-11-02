from miramace.geo import ZIP_CENTROIDS, distance_mi


def test_directory_zips_present():
    for z in ["60614", "60654", "60618", "60640", "60605", "60608", "60657", "60714", "60201"]:
        assert z in ZIP_CENTROIDS


def test_distance_basic():
    assert distance_mi("60614", "60614") == 0.0
    d = distance_mi("60614", "60657")
    assert 0.5 < d < 3
    assert distance_mi("60614", "60201") > distance_mi("60614", "60657")
    assert distance_mi("60614", "60657") == distance_mi("60657", "60614")


def test_unknown_zip_fallback():
    assert distance_mi("60614", "60699") == abs(60614 - 60699) * 0.5
