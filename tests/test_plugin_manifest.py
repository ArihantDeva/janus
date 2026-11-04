"""Native Muse plugin manifest: janus-failure-lens ships valid, installed, approved."""

import json
import os

PLUGIN_DIR = os.path.join(os.path.dirname(__file__), "..", "plugins",
                          "janus-failure-lens")


def read_manifest():
    path = os.path.join(PLUGIN_DIR, ".muse-plugin", "plugin.json")
    return json.loads(open(path).read())


def test_plugin_manifest_exists_and_names_janus():
    manifest = read_manifest()
    assert manifest["name"] == "janus-failure-lens"
    assert manifest["schemaVersion"] == 1
    assert manifest["compat"]["manifestDir"] == ".muse-plugin"


def test_plugin_manifest_declares_post_failure_hook():
    manifest = read_manifest()
    hooks = manifest["capabilities"]["hooks"]
    events = [h["event"] for h in hooks]
    assert "PostToolUseFailure" in events
