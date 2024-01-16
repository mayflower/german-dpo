import debugpy
from flask import Flask
from flask.cli import FlaskGroup
import scripts.openai_translate as ot

def create_app():
    app = Flask(__name__)
    return app

cli = FlaskGroup(create_app=create_app)

@cli.command('estimate_total_cost')
def estimate_total_cost():
  """
  debugpy.listen(("0.0.0.0", 5678))
  print("Waiting for client to attach...")
  debugpy.wait_for_client()
  """
  ot.estimate_total_cost()

if __name__ == "__main__":
  cli()