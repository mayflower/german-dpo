import debugpy
from flask import Flask
from flask.cli import FlaskGroup
from scripts import seed_db

def create_app():
    app = Flask(__name__)
    return app

cli = FlaskGroup(create_app=create_app)

@cli.command('run')
def run():
  """
  debugpy.listen(("0.0.0.0", 5678))
  print("Waiting for client to attach...")
  debugpy.wait_for_client()
  """



if __name__ == "__main__":
  cli()