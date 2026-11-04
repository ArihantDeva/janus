import {
  AbsoluteFill,
  Sequence,
  Video,
  interpolate,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";

// V2 RULE: every terminal pixel comes from a real capture (staticFile MP4s,
// OCR-verified). Remotion renders ONLY cards, transitions, PiP framing.
// No typeset fake terminals anywhere in this composition.

const BG = "#0A0C10";
const GOLD = "#D4AF69";
const DIM = "#969EAC";
const TEXT = "#E8ECF2";
const MONO = "Menlo, Monaco, monospace";

const Fade: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const opacity = interpolate(
    frame,
    [0, 12, durationInFrames - 12, durationInFrames],
    [0, 1, 1, 0],
  );
  return <AbsoluteFill style={{ opacity }}>{children}</AbsoluteFill>;
};

const TitleCard: React.FC<{ title: string; sub: string }> = ({ title, sub }) => (
  <AbsoluteFill
    style={{
      backgroundColor: BG,
      justifyContent: "center",
      alignItems: "center",
    }}
  >
    <div
      style={{
        fontFamily: MONO,
        fontSize: 64,
        color: GOLD,
        letterSpacing: 24,
      }}
    >
      {title}
    </div>
    <div style={{ fontFamily: MONO, fontSize: 26, color: DIM, marginTop: 20 }}>
      {sub}
    </div>
  </AbsoluteFill>
);

const Numbers: React.FC = () => {
  const frame = useCurrentFrame();
  const rows = [
    ["82.20%", "teacher agreement"],
    ["+27.0 pts", "over majority floor"],
    ["0.10", "calibration error"],
    ["90% @ 83.4%", "answered when sure"],
  ];
  return (
    <AbsoluteFill
      style={{ backgroundColor: BG, justifyContent: "center", paddingLeft: 220 }}
    >
      {rows.map(([v, l], i) => {
        const local = frame - i * 12;
        const opacity = interpolate(local, [0, 10], [0, 1], {
          extrapolateLeft: "clamp",
          extrapolateRight: "clamp",
        });
        const dx = interpolate(local, [0, 10], [30, 0], {
          extrapolateLeft: "clamp",
          extrapolateRight: "clamp",
        });
        return (
          <div
            key={l}
            style={{
              opacity,
              transform: `translateX(${dx}px)`,
              fontFamily: MONO,
              marginBottom: 28,
            }}
          >
            <span style={{ fontSize: 56, color: GOLD }}>{v}</span>
            <span style={{ fontSize: 28, color: DIM, marginLeft: 24 }}>{l}</span>
          </div>
        );
      })}
    </AbsoluteFill>
  );
};

const Caption: React.FC<{ text: string }> = ({ text }) => (
  <div
    style={{
      position: "absolute",
      bottom: 60,
      width: "100%",
      textAlign: "center",
      fontFamily: MONO,
      fontSize: 30,
      color: TEXT,
      textShadow: "0 2px 12px #000",
    }}
  >
    {text}
  </div>
);

export const FPS = 30;

// Beat plan (narration master, 105.7s). Real footage beats use <Video> over
// demo/final captures; card beats use rendered graphics only.
const BEATS = [
  { name: "Title", dur: 3, el: <TitleCard title="JANUS" sub="a System-1 reflex under a System-2 agent" /> },
  {
    name: "B1-hook", dur: 11.59,
    el: (
      <>
        <Video src={staticFile("face_b1.mp4")} style={{ width: "100%", height: "100%", objectFit: "cover" }} />
        <Caption text="Every day, frontier coding agents burn their tokens on the simplest failure mode: tool calls." />
      </>
    ),
  },
  {
    name: "B2-failure", dur: 17.14,
    el: (
      <>
        <Video src={staticFile("face_b2.mp4")} style={{ width: "100%", height: "100%", objectFit: "cover" }} />
        <Caption text="The command fails. No signal about why — 800 reasoning tokens and 4 seconds to diagnose." />
      </>
    ),
  },
  {
    name: "B3-verdict", dur: 15.77,
    el: (
      <>
        <Video
          src={staticFile("verdict.mp4")}
          loop
          style={{ width: "100%", height: "100%", objectFit: "contain", backgroundColor: BG }}
        />
        <Caption text="Janus: fix_input, confidence one point zero — from a 1B model next to the workload." />
      </>
    ),
  },
  {
    name: "B4-cost", dur: 19.32,
    el: (
      <>
        <Video src={staticFile("face_b4.mp4")} style={{ width: "100%", height: "100%", objectFit: "cover" }} />
        <div style={{ position: "absolute", right: 40, bottom: 120, width: 640, borderRadius: 12, overflow: "hidden", border: "2px solid #D4AF69" }}>
          <Video src={staticFile("race.mp4")} loop style={{ width: "100%" }} />
        </div>
        <Caption text="Same question: 4.1s and billing vs 9.4ms on GPU and free." />
      </>
    ),
  },
  { name: "Agree", dur: 5, el: <TitleCard title="THEY AGREE ON HALF" sub="where they disagree, janus escalates" /> },
  {
    name: "B5-why", dur: 27.10,
    el: (
      <>
        <Video src={staticFile("face_b5.mp4")} style={{ width: "100%", height: "100%", objectFit: "cover" }} />
        <Caption text="One forward pass. Three answers. Code, not the model, makes the final call." />
      </>
    ),
  },
  { name: "Nums", dur: 6, el: <Numbers /> },
  {
    name: "B6-close", dur: 14.74,
    el: (
      <>
        <Video src={staticFile("face_b6.mp4")} style={{ width: "100%", height: "100%", objectFit: "cover" }} />
        <Caption text="Janus is a System-1 guard under a System-2 agent." />
      </>
    ),
  },
  { name: "End", dur: 3, el: <TitleCard title="JANUS" sub="install it, break something, watch it answer" /> },
];

let acc = 0;
const TIMED = BEATS.map((b) => {
  const from = Math.round(acc * FPS);
  acc += b.dur;
  return { ...b, from, len: Math.round(b.dur * FPS) };
});
const TOTAL = Math.round(acc * FPS);

export const JANUS_FPS = FPS;
export const JANUS_TOTAL = TOTAL;

export const JanusV2: React.FC = () => {
  return (
    <>
      {TIMED.map((b, i) => (
        <Sequence key={i} from={b.from} durationInFrames={b.len} name={b.name}>
          <Fade>{b.el}</Fade>
        </Sequence>
      ))}
    </>
  );
};
