# Janus A/B: same failure, without vs with the hook.
# Both panes are REAL asciinema captures (demo/ab/*.cast), not synthesized
# footage. VHS only composites + letterboxes; every terminal frame is a
# recorded byte stream. Anthropic-approximate palette: warm dark ground,
# coral accent, cream text.
Output demo/ab/janus_ab.mp4
Require mise

# --- palette: warm paper-on-ink, coral accent (Anthropic-approx) ---
Set Theme {"background": "#1F1E1D", "foreground": "#F0EEE6", "cursor": "#CC785C", "selection": "#CC785C", "black": "#1F1E1D", "red": "#CC785C", "green": "#7FB069", "yellow": "#E5C07B", "blue": "#7AA2F7", "magenta": "#C678DD", "cyan": "#56B6C2", "white": "#F0EEE6", "brightBlack": "#5C5752", "brightRed": "#E09879", "brightGreen": "#98C379", "brightYellow": "#E5C07B", "brightBlue": "#7AA2F7", "brightMagenta": "#C678DD", "brightCyan": "#56B6C2", "brightWhite": "#F0EEE6"}
Set FontSize 18
Set Width 1280
Set Height 720
Set Padding 24
Set Framerate 30

Hide
Type "WITHOUT Janus — no advisory possible" Enter
Show

Sleep 1s
Type "mise exec -- uvx --from asciinema asciinema play demo/ab/without.cast" Enter
Sleep 17s
Ctrl+C

Sleep 1s
Hide
Type "WITH Janus — hook advisory lands" Enter
Show

Sleep 1s
Type "mise exec -- uvx --from asciinema asciinema play demo/ab/with.cast" Enter
Sleep 31s
Ctrl+C

Sleep 1s
Type "echo 'same failure. one reflex apart.'" Enter
Sleep 2s
