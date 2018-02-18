import offshoot


class SerpentSuperflightGameAgentPlugin(offshoot.Plugin):
    name = "SerpentSuperflightGameAgentPlugin"
    version = "0.1.0"

    plugins = []

    libraries = [
        "tensorforce==0.3.5.1"
    ]

    files = [
        {"path": "serpent_Superflight_game_agent.py", "pluggable": "GameAgent"}
    ]

    config = {
        "frame_handler": "PLAY"
    }

    @classmethod
    def on_install(cls):
        print("\n\n%s was installed successfully!" % cls.__name__)

    @classmethod
    def on_uninstall(cls):
        print("\n\n%s was uninstalled successfully!" % cls.__name__)


if __name__ == "__main__":
    offshoot.executable_hook(SerpentSuperflightGameAgentPlugin)
