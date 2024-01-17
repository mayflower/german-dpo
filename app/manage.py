import debugpy
from flask import Flask
from flask.cli import FlaskGroup
import scripts.openai_translate as ot
import scripts.inference as inf

def create_app():
    app = Flask(__name__)
    return app

cli = FlaskGroup(create_app=create_app)

@cli.command('estimate_translation')
def estimate_translation():
  ot.estimate_translation()

@cli.command('estimate_inference')
def estimate_inference():
  #"""
  debugpy.listen(("0.0.0.0", 5678))
  print("Waiting for client to attach...")
  debugpy.wait_for_client()
  #"""
  inf.estimate_inference()

@cli.command('run_translation')
def run_translation():
  ot.run_translation()

if __name__ == "__main__":
  cli()