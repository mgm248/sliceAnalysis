
def test_more_app(tmp_path):
    from more_kivy_app.app import MoreKivyApp
    config_file = tmp_path / 'config.yaml'

    class MyApp(MoreKivyApp):
        yaml_config_path = str(config_file)

    app = MyApp()
    assert config_file.exists()
