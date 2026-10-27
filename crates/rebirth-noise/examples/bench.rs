//! cargo run --release --example bench -p rebirth-noise
use rebirth_noise::*;
use std::time::Instant;

fn main() {
    let (w, h, d) = (128usize, 128, 128);
    let mut buf = vec![0.0f32; w * h * d];
    println!("объём {w}x{h}x{d} = {} значений, 1 октава", buf.len());
    for kind in [Kind::Value, Kind::Perlin, Kind::Simplex, Kind::Cellular] {
        let mut n = Noise::new(kind, 42);
        n.octaves = 1;
        n.frequency = 0.05;
        let t = Instant::now();
        n.fill3(&mut buf, w, h, d, (0.0, 0.0, 0.0), 1.0);
        let dt = t.elapsed().as_secs_f64();
        let (mn, mx) = buf
            .iter()
            .fold((f32::MAX, f32::MIN), |a, &v| (a.0.min(v), a.1.max(v)));
        let mean = buf.iter().map(|&v| v as f64).sum::<f64>() / buf.len() as f64;
        let clipped =
            buf.iter().filter(|&&v| v.abs() >= 0.9999).count() as f64 / buf.len() as f64 * 100.0;
        println!("{kind:?}: {:.1} Мзнач/с, диапазон {mn:.3}..{mx:.3}, среднее {mean:.3}, на границе ±1: {clipped:.2}%", buf.len() as f64 / dt / 1e6);
    }
    let mut n = Noise::new(Kind::Simplex, 42);
    n.octaves = 5;
    n.frequency = 0.02;
    let t = Instant::now();
    n.fill3(&mut buf, w, h, d, (0.0, 0.0, 0.0), 1.0);
    println!(
        "Simplex fBm 5 октав: {:.1} Мзнач/с",
        buf.len() as f64 / t.elapsed().as_secs_f64() / 1e6
    );
}
