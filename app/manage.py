import debugpy
from flask import Flask
from flask.cli import FlaskGroup
import scripts.openai_translate as ot

def create_app():
    app = Flask(__name__)
    return app

cli = FlaskGroup(create_app=create_app)

@cli.command('estimate_metrics')
def estimate_metrics():
  ot.estimate_metrics()

@cli.command('run_translations')
def run_translations():
  #"""
  debugpy.listen(("0.0.0.0", 5678))
  print("Waiting for client to attach...")
  debugpy.wait_for_client()
  #"""
  ot.run_translations()

if __name__ == "__main__":
  cli()