//! Когерентный шум на безопасном Rust: значимостный (value), Перлина, симплексный и клеточный (Ворлея),
//! фрактальные комбинаторы (fBm, billow, rigid) и заполнение 3D-объёмов.
//!
//! Координаты хэшируются арифметикой (умножения и сдвиги), без таблиц перестановок: нет обращений к памяти
//! и ничего не нужно инициализировать. Результат зависит только от входа и зерна и совпадает на всех платформах.
//! Идея навеяна FastNoiseSIMD Джордана Пека (MIT); код написан с нуля, без SIMD, на скалярной арифметике.
#![forbid(unsafe_code)]

#[inline]
fn hash(seed: u32, x: i32, y: i32, z: i32) -> u32 {
    let mut h = seed ^ 0x9E37_79B9;
    h = (h ^ (x as u32).wrapping_mul(0x27D4_EB2D)).wrapping_mul(0x85EB_CA6B);
    h = (h ^ (y as u32).wrapping_mul(0x1656_67B1)).wrapping_mul(0xC2B2_AE35);
    h = (h ^ (z as u32).wrapping_mul(0x9E37_79B1)).wrapping_mul(0x27D4_EB2F);
    h ^= h >> 15;
    h = h.wrapping_mul(0x85EB_CA77);
    h ^= h >> 13;
    h = h.wrapping_mul(0xC2B2_AE3D);
    h ^ (h >> 16)
}

/// Хэш в `[-1, 1]`.
#[inline]
fn hash_unit(h: u32) -> f32 {
    (h >> 8) as f32 * (2.0 / 16_777_216.0) - 1.0
}

/// Хэш в `[0, 1)`.
#[inline]
fn hash_01(h: u32) -> f32 {
    (h >> 8) as f32 * (1.0 / 16_777_216.0)
}

#[inline]
fn floor_i(x: f32) -> i32 {
    x.floor() as i32
}

#[inline]
fn quintic(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Градиенты Перлина в 3D: 12 рёбер куба (по Перлину 2002), для хэша берётся `h % 12`; 4 запасных повтора.
#[inline]
fn grad3(h: u32, x: f32, y: f32, z: f32) -> f32 {
    match h % 12 {
        0 => x + y,
        1 => -x + y,
        2 => x - y,
        3 => -x - y,
        4 => x + z,
        5 => -x + z,
        6 => x - z,
        7 => -x - z,
        8 => y + z,
        9 => -y + z,
        10 => y - z,
        _ => -y - z,
    }
}

/// Градиенты в 2D: 8 направлений (4 оси и 4 диагонали).
#[inline]
fn grad2(h: u32, x: f32, y: f32) -> f32 {
    match h & 7 {
        0 => x,
        1 => -x,
        2 => y,
        3 => -y,
        4 => x + y,
        5 => x - y,
        6 => -x + y,
        _ => -x - y,
    }
}

/// Что возвращает клеточный шум.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CellularReturn {
    /// Расстояние до ближайшего центра (F1).
    Distance,
    /// Расстояние до второго по близости (F2).
    Distance2,
    /// F1 + F2.
    Distance2Add,
    /// F2 − F1: яркие границы ячеек.
    Distance2Sub,
    /// F1 · F2.
    Distance2Mul,
    /// Случайное значение ячейки (идентификатор), одинаковое внутри ячейки.
    Value,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CellularDistance {
    Euclidean,
    Manhattan,
    Chebyshev,
}

/// Способ складывать октавы.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Fractal {
    /// Сумма с затухающим весом: fBm.
    Fbm,
    /// 2·|n| − 1: пухлые складки.
    Billow,
    /// (1 − |n|)², с обратной связью между октавами: острые горные хребты.
    Rigid,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    Value,
    Perlin,
    Simplex,
    Cellular,
}

/// Настроенный генератор. Все методы чистые: одно и то же зерно и координаты дают один и тот же ответ.
#[derive(Clone, Copy, Debug)]
pub struct Noise {
    pub seed: u32,
    pub frequency: f32,
    pub kind: Kind,
    pub fractal: Fractal,
    pub octaves: u32,
    pub lacunarity: f32,
    pub gain: f32,
    pub cellular_return: CellularReturn,
    pub cellular_distance: CellularDistance,
    /// Насколько центр ячейки сдвинут от решётки (0 — правильная сетка, 1 — максимально хаотично).
    pub cellular_jitter: f32,
}

impl Default for Noise {
    fn default() -> Self {
        Noise {
            seed: 1337,
            frequency: 0.01,
            kind: Kind::Simplex,
            fractal: Fractal::Fbm,
            octaves: 4,
            lacunarity: 2.0,
            gain: 0.5,
            cellular_return: CellularReturn::Distance,
            cellular_distance: CellularDistance::Euclidean,
            cellular_jitter: 1.0,
        }
    }
}

impl Noise {
    pub fn new(kind: Kind, seed: u32) -> Self {
        Noise {
            kind,
            seed,
            ..Noise::default()
        }
    }

    // ---------- базовый шум, значения в [-1, 1] ----------

    pub fn value3(&self, x: f32, y: f32, z: f32, seed: u32) -> f32 {
        let (xi, yi, zi) = (floor_i(x), floor_i(y), floor_i(z));
        let (u, v, w) = (
            quintic(x - xi as f32),
            quintic(y - yi as f32),
            quintic(z - zi as f32),
        );
        let h = |dx, dy, dz| hash_unit(hash(seed, xi + dx, yi + dy, zi + dz));
        let x0 = lerp(
            lerp(h(0, 0, 0), h(1, 0, 0), u),
            lerp(h(0, 1, 0), h(1, 1, 0), u),
            v,
        );
        let x1 = lerp(
            lerp(h(0, 0, 1), h(1, 0, 1), u),
            lerp(h(0, 1, 1), h(1, 1, 1), u),
            v,
        );
        lerp(x0, x1, w)
    }

    pub fn value2(&self, x: f32, y: f32, seed: u32) -> f32 {
        let (xi, yi) = (floor_i(x), floor_i(y));
        let (u, v) = (quintic(x - xi as f32), quintic(y - yi as f32));
        let h = |dx, dy| hash_unit(hash(seed, xi + dx, yi + dy, 0));
        lerp(lerp(h(0, 0), h(1, 0), u), lerp(h(0, 1), h(1, 1), u), v)
    }

    pub fn perlin3(&self, x: f32, y: f32, z: f32, seed: u32) -> f32 {
        let (xi, yi, zi) = (floor_i(x), floor_i(y), floor_i(z));
        let (fx, fy, fz) = (x - xi as f32, y - yi as f32, z - zi as f32);
        let (u, v, w) = (quintic(fx), quintic(fy), quintic(fz));
        let g = |dx: i32, dy: i32, dz: i32| {
            grad3(
                hash(seed, xi + dx, yi + dy, zi + dz),
                fx - dx as f32,
                fy - dy as f32,
                fz - dz as f32,
            )
        };
        let x0 = lerp(
            lerp(g(0, 0, 0), g(1, 0, 0), u),
            lerp(g(0, 1, 0), g(1, 1, 0), u),
            v,
        );
        let x1 = lerp(
            lerp(g(0, 0, 1), g(1, 0, 1), u),
            lerp(g(0, 1, 1), g(1, 1, 1), u),
            v,
        );
        // эмпирически сырой максимум около 0,9; множитель растягивает диапазон почти до единицы
        (lerp(x0, x1, w) * 1.1).clamp(-1.0, 1.0)
    }

    pub fn perlin2(&self, x: f32, y: f32, seed: u32) -> f32 {
        let (xi, yi) = (floor_i(x), floor_i(y));
        let (fx, fy) = (x - xi as f32, y - yi as f32);
        let (u, v) = (quintic(fx), quintic(fy));
        let g = |dx: i32, dy: i32| {
            grad2(
                hash(seed, xi + dx, yi + dy, 0),
                fx - dx as f32,
                fy - dy as f32,
            )
        };
        (lerp(lerp(g(0, 0), g(1, 0), u), lerp(g(0, 1), g(1, 1), u), v) * 1.1).clamp(-1.0, 1.0)
    }

    pub fn simplex2(&self, x: f32, y: f32, seed: u32) -> f32 {
        const F2: f32 = 0.366_025_4; // (√3 − 1) / 2
        const G2: f32 = 0.211_324_87; // (3 − √3) / 6
        let s = (x + y) * F2;
        let (i, j) = (floor_i(x + s), floor_i(y + s));
        let t = (i + j) as f32 * G2;
        let (x0, y0) = (x - (i as f32 - t), y - (j as f32 - t));
        let (i1, j1) = if x0 > y0 { (1, 0) } else { (0, 1) };
        let (x1, y1) = (x0 - i1 as f32 + G2, y0 - j1 as f32 + G2);
        let (x2, y2) = (x0 - 1.0 + 2.0 * G2, y0 - 1.0 + 2.0 * G2);
        let corner = |dx: f32, dy: f32, ci: i32, cj: i32| {
            let t = 0.5 - dx * dx - dy * dy;
            if t < 0.0 {
                0.0
            } else {
                let t2 = t * t;
                t2 * t2 * grad2(hash(seed, ci, cj, 0), dx, dy)
            }
        };
        let n =
            corner(x0, y0, i, j) + corner(x1, y1, i + i1, j + j1) + corner(x2, y2, i + 1, j + 1);
        (n * 45.0).clamp(-1.0, 1.0)
    }

    pub fn simplex3(&self, x: f32, y: f32, z: f32, seed: u32) -> f32 {
        const F3: f32 = 1.0 / 3.0;
        const G3: f32 = 1.0 / 6.0;
        let s = (x + y + z) * F3;
        let (i, j, k) = (floor_i(x + s), floor_i(y + s), floor_i(z + s));
        let t = (i + j + k) as f32 * G3;
        let (x0, y0, z0) = (x - (i as f32 - t), y - (j as f32 - t), z - (k as f32 - t));
        let ((i1, j1, k1), (i2, j2, k2)) = if x0 >= y0 {
            if y0 >= z0 {
                ((1, 0, 0), (1, 1, 0))
            } else if x0 >= z0 {
                ((1, 0, 0), (1, 0, 1))
            } else {
                ((0, 0, 1), (1, 0, 1))
            }
        } else if y0 < z0 {
            ((0, 0, 1), (0, 1, 1))
        } else if x0 < z0 {
            ((0, 1, 0), (0, 1, 1))
        } else {
            ((0, 1, 0), (1, 1, 0))
        };
        let (x1, y1, z1) = (
            x0 - i1 as f32 + G3,
            y0 - j1 as f32 + G3,
            z0 - k1 as f32 + G3,
        );
        let (x2, y2, z2) = (
            x0 - i2 as f32 + 2.0 * G3,
            y0 - j2 as f32 + 2.0 * G3,
            z0 - k2 as f32 + 2.0 * G3,
        );
        let (x3, y3, z3) = (
            x0 - 1.0 + 3.0 * G3,
            y0 - 1.0 + 3.0 * G3,
            z0 - 1.0 + 3.0 * G3,
        );
        let corner = |dx: f32, dy: f32, dz: f32, ci: i32, cj: i32, ck: i32| {
            let t = 0.6 - dx * dx - dy * dy - dz * dz;
            if t < 0.0 {
                0.0
            } else {
                let t2 = t * t;
                t2 * t2 * grad3(hash(seed, ci, cj, ck), dx, dy, dz)
            }
        };
        let n = corner(x0, y0, z0, i, j, k)
            + corner(x1, y1, z1, i + i1, j + j1, k + k1)
            + corner(x2, y2, z2, i + i2, j + j2, k + k2)
            + corner(x3, y3, z3, i + 1, j + 1, k + 1);
        (n * 32.0).clamp(-1.0, 1.0)
    }

    /// Клеточный шум в 3D. Результат приведён примерно к `[-1, 1]`.
    pub fn cellular3(&self, x: f32, y: f32, z: f32, seed: u32) -> f32 {
        let (xi, yi, zi) = (floor_i(x), floor_i(y), floor_i(z));
        let (mut f1, mut f2) = (f32::INFINITY, f32::INFINITY);
        let mut id = 0u32;
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (cx, cy, cz) = (xi + dx, yi + dy, zi + dz);
                    let h = hash(seed, cx, cy, cz);
                    let j = self.cellular_jitter;
                    let px = cx as f32 + 0.5 + (hash_01(h) - 0.5) * j;
                    let py = cy as f32 + 0.5 + (hash_01(h.rotate_left(11) ^ 0x68E3_1DA4) - 0.5) * j;
                    let pz = cz as f32 + 0.5 + (hash_01(h.rotate_left(22) ^ 0xB529_7A4D) - 0.5) * j;
                    let (ax, ay, az) = ((px - x).abs(), (py - y).abs(), (pz - z).abs());
                    let d = match self.cellular_distance {
                        CellularDistance::Euclidean => (ax * ax + ay * ay + az * az).sqrt(),
                        CellularDistance::Manhattan => ax + ay + az,
                        CellularDistance::Chebyshev => ax.max(ay).max(az),
                    };
                    if d < f1 {
                        f2 = f1;
                        f1 = d;
                        id = h;
                    } else if d < f2 {
                        f2 = d;
                    }
                }
            }
        }
        // F1 при единичном джиттере не превосходит ~√3/2 по ячейке, F2 до ~√3; нормируем под диапазон
        let r = match self.cellular_return {
            CellularReturn::Distance => f1 * 2.0 - 1.0,
            CellularReturn::Distance2 => f2 - 1.0,
            CellularReturn::Distance2Add => (f1 + f2) - 1.0,
            CellularReturn::Distance2Sub => (f2 - f1) * 2.0 - 1.0,
            CellularReturn::Distance2Mul => f1 * f2 * 2.0 - 1.0,
            CellularReturn::Value => hash_unit(id),
        };
        r.clamp(-1.0, 1.0)
    }

    pub fn cellular2(&self, x: f32, y: f32, seed: u32) -> f32 {
        self.cellular3(x, y, 0.5, seed)
    }

    fn base3(&self, x: f32, y: f32, z: f32, seed: u32) -> f32 {
        match self.kind {
            Kind::Value => self.value3(x, y, z, seed),
            Kind::Perlin => self.perlin3(x, y, z, seed),
            Kind::Simplex => self.simplex3(x, y, z, seed),
            Kind::Cellular => self.cellular3(x, y, z, seed),
        }
    }

    fn base2(&self, x: f32, y: f32, seed: u32) -> f32 {
        match self.kind {
            Kind::Value => self.value2(x, y, seed),
            Kind::Perlin => self.perlin2(x, y, seed),
            Kind::Simplex => self.simplex2(x, y, seed),
            Kind::Cellular => self.cellular2(x, y, seed),
        }
    }

    // ---------- фрактальные комбинации ----------

    /// Сумма амплитуд всех октав: нужна, чтобы результат остался в `[-1, 1]`.
    fn amp_sum(&self) -> f32 {
        let (mut a, mut s) = (1.0, 0.0);
        for _ in 0..self.octaves.max(1) {
            s += a;
            a *= self.gain;
        }
        s
    }

    /// Шум в точке с фракталом, результат в `[-1, 1]`.
    pub fn get3(&self, x: f32, y: f32, z: f32) -> f32 {
        let (mut x, mut y, mut z) = (x * self.frequency, y * self.frequency, z * self.frequency);
        let mut seed = self.seed;
        let (mut amp, mut sum) = (1.0f32, 0.0f32);
        let mut weight = 1.0f32;
        for _ in 0..self.octaves.max(1) {
            let n = self.base3(x, y, z, seed);
            match self.fractal {
                Fractal::Fbm => sum += n * amp,
                Fractal::Billow => sum += (n.abs() * 2.0 - 1.0) * amp,
                Fractal::Rigid => {
                    let r = (1.0 - n.abs()).powi(2) * 2.0 - 1.0;
                    sum += r * amp * weight;
                    weight = (r * 0.5 + 0.5).clamp(0.0, 1.0);
                }
            }
            x *= self.lacunarity;
            y *= self.lacunarity;
            z *= self.lacunarity;
            amp *= self.gain;
            seed = seed.wrapping_add(1);
        }
        (sum / self.amp_sum()).clamp(-1.0, 1.0)
    }

    pub fn get2(&self, x: f32, y: f32) -> f32 {
        let (mut x, mut y) = (x * self.frequency, y * self.frequency);
        let mut seed = self.seed;
        let (mut amp, mut sum) = (1.0f32, 0.0f32);
        let mut weight = 1.0f32;
        for _ in 0..self.octaves.max(1) {
            let n = self.base2(x, y, seed);
            match self.fractal {
                Fractal::Fbm => sum += n * amp,
                Fractal::Billow => sum += (n.abs() * 2.0 - 1.0) * amp,
                Fractal::Rigid => {
                    let r = (1.0 - n.abs()).powi(2) * 2.0 - 1.0;
                    sum += r * amp * weight;
                    weight = (r * 0.5 + 0.5).clamp(0.0, 1.0);
                }
            }
            x *= self.lacunarity;
            y *= self.lacunarity;
            amp *= self.gain;
            seed = seed.wrapping_add(1);
        }
        (sum / self.amp_sum()).clamp(-1.0, 1.0)
    }

    /// Заполняет плоский буфер `w × h` (порядок строк), левый верхний угол в `(x0, y0)`.
    pub fn fill2(&self, out: &mut [f32], w: usize, h: usize, x0: f32, y0: f32, step: f32) {
        assert!(out.len() >= w * h);
        for j in 0..h {
            let y = y0 + j as f32 * step;
            for i in 0..w {
                out[j * w + i] = self.get2(x0 + i as f32 * step, y);
            }
        }
    }

    /// Заполняет объём `w × h × d` (x меняется быстрее всего).
    pub fn fill3(
        &self,
        out: &mut [f32],
        w: usize,
        h: usize,
        d: usize,
        origin: (f32, f32, f32),
        step: f32,
    ) {
        assert!(out.len() >= w * h * d);
        for k in 0..d {
            let z = origin.2 + k as f32 * step;
            for j in 0..h {
                let y = origin.1 + j as f32 * step;
                for i in 0..w {
                    out[(k * h + j) * w + i] = self.get3(origin.0 + i as f32 * step, y, z);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_kinds() -> [Kind; 4] {
        [Kind::Value, Kind::Perlin, Kind::Simplex, Kind::Cellular]
    }

    #[test]
    fn deterministic_and_seed_sensitive() {
        for k in all_kinds() {
            let a = Noise::new(k, 1);
            let b = Noise::new(k, 1);
            let c = Noise::new(k, 2);
            let mut diff = 0;
            for i in 0..200 {
                let (x, y, z) = (i as f32 * 3.7, i as f32 * 1.3, i as f32 * 0.9);
                assert_eq!(a.get3(x, y, z), b.get3(x, y, z));
                if a.get3(x, y, z) != c.get3(x, y, z) {
                    diff += 1;
                }
            }
            assert!(diff > 100, "{k:?}: зерно почти не влияет");
        }
    }

    #[test]
    fn stays_in_range_and_has_variance() {
        for k in all_kinds() {
            for f in [Fractal::Fbm, Fractal::Billow, Fractal::Rigid] {
                let mut n = Noise::new(k, 5);
                n.fractal = f;
                n.frequency = 0.07;
                let (mut lo, mut hi, mut sum, mut sum2, cnt) =
                    (2.0f32, -2.0f32, 0.0f64, 0.0f64, 20_000);
                for i in 0..cnt {
                    let v = n.get3(
                        (i % 100) as f32 * 1.1,
                        ((i / 100) % 100) as f32 * 0.9,
                        (i / 10000) as f32 * 3.0,
                    );
                    assert!(
                        v.is_finite() && (-1.0..=1.0).contains(&v),
                        "{k:?} {f:?}: {v}"
                    );
                    lo = lo.min(v);
                    hi = hi.max(v);
                    sum += v as f64;
                    sum2 += (v * v) as f64;
                }
                let mean = sum / cnt as f64;
                let var = sum2 / cnt as f64 - mean * mean;
                assert!(
                    var > 0.005,
                    "{k:?} {f:?}: шум почти постоянный (дисперсия {var}), диапазон {lo}..{hi}"
                );
            }
        }
    }

    #[test]
    fn smooth_for_continuous_kinds() {
        // Малый шаг даёт малое изменение (клеточный шум разрывен на границах, его не проверяем)
        for k in [Kind::Value, Kind::Perlin, Kind::Simplex] {
            let mut n = Noise::new(k, 9);
            n.octaves = 1;
            n.frequency = 1.0;
            let mut maxd = 0.0f32;
            for i in 0..5000 {
                let (x, y, z) = (i as f32 * 0.137, i as f32 * 0.071, i as f32 * 0.053);
                maxd = maxd.max((n.get3(x, y, z) - n.get3(x + 0.001, y, z)).abs());
            }
            assert!(maxd < 0.02, "{k:?}: скачок {maxd}");
        }
    }

    #[test]
    fn perlin_is_zero_on_lattice() {
        let n = Noise::new(Kind::Perlin, 3);
        for (x, y, z) in [(0, 0, 0), (3, -2, 7), (-5, 4, 1)] {
            assert_eq!(n.perlin3(x as f32, y as f32, z as f32, 3), 0.0);
            assert_eq!(n.perlin2(x as f32, y as f32, 3), 0.0);
        }
    }

    #[test]
    fn cellular_f1_le_f2_and_value_constant_in_cell() {
        let mut n = Noise::new(Kind::Cellular, 4);
        n.octaves = 1;
        n.frequency = 1.0;
        for i in 0..2000 {
            let (x, y, z) = (i as f32 * 0.31, i as f32 * 0.17, i as f32 * 0.11);
            n.cellular_return = CellularReturn::Distance;
            let a = n.cellular3(x, y, z, 4);
            n.cellular_return = CellularReturn::Distance2;
            let b = n.cellular3(x, y, z, 4);
            // F1·2−1 и F2−1: сравниваем в исходных единицах
            assert!((a + 1.0) / 2.0 <= b + 1.0 + 1e-5);
        }
        n.cellular_return = CellularReturn::Value;
        // две соседние точки рядом с центром одной ячейки видят одно и то же значение
        let v1 = n.cellular3(10.5, 10.5, 10.5, 4);
        let v2 = n.cellular3(10.5001, 10.5, 10.5, 4);
        assert_eq!(v1, v2);
    }

    #[test]
    fn jitter_zero_gives_regular_grid() {
        let mut n = Noise::new(Kind::Cellular, 4);
        n.cellular_jitter = 0.0;
        n.octaves = 1;
        n.frequency = 1.0;
        // центр ячейки (0.5, 0.5, 0.5): расстояние нулевое → результат −1
        assert!((n.cellular3(0.5, 0.5, 0.5, 4) + 1.0).abs() < 1e-5);
    }

    #[test]
    fn fill_matches_pointwise() {
        let n = Noise::new(Kind::Simplex, 6);
        let (w, h, d) = (5, 4, 3);
        let mut buf = vec![0.0; w * h * d];
        n.fill3(&mut buf, w, h, d, (1.0, 2.0, 3.0), 0.5);
        for k in 0..d {
            for j in 0..h {
                for i in 0..w {
                    assert_eq!(
                        buf[(k * h + j) * w + i],
                        n.get3(
                            1.0 + i as f32 * 0.5,
                            2.0 + j as f32 * 0.5,
                            3.0 + k as f32 * 0.5
                        )
                    );
                }
            }
        }
        let mut b2 = vec![0.0; 20];
        n.fill2(&mut b2, 5, 4, 0.0, 0.0, 1.0);
        assert_eq!(b2[7], n.get2(2.0, 1.0));
    }

    #[test]
    fn negative_and_large_coordinates_are_fine() {
        for k in all_kinds() {
            let n = Noise::new(k, 1);
            for &(x, y, z) in &[
                (-1e5f32, 3e4, -7.7),
                (0.0, 0.0, 0.0),
                (-0.0001, -0.0001, -0.0001),
                (1e6, -1e6, 1e6),
            ] {
                let v = n.get3(x, y, z);
                assert!(v.is_finite() && v.abs() <= 1.0);
            }
        }
    }
}
