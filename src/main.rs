use anyhow::{Context, Result};
use clap::Parser;
use image::imageops::resize;
use image::imageops::FilterType::Lanczos3;
use image::{ImageBuffer, Rgb};
use indicatif::{ProgressBar, ProgressStyle};
use jwalk::WalkDir;
use ndarray::Array2;
use rayon::prelude::*;
use rustfft::{num_complex::Complex, FftPlanner};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_MP3};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    input_dir: String,

    #[arg(short, long)]
    output_dir: String,

    #[arg(long, default_value_t = 1536)]
    fft_size: usize,

    #[arg(long, default_value_t = 768)]
    hop_size: usize,

    #[arg(long, default_value = "png")]
    output_format: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Create output directory if it doesn't exist
    std::fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("Failed to create output directory: {}", args.output_dir))?;

    // Get all MP3 files in the input directory
    let mp3_files: Vec<PathBuf> = WalkDir::new(&args.input_dir)
        .into_iter()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().extension().map_or(false, |ext| ext == "mp3"))
        .map(|entry| entry.path().to_owned())
        .collect();

    // Set up progress bar
    let progress_bar = Arc::new(ProgressBar::with_draw_target(
        Some(mp3_files.len() as u64),
        indicatif::ProgressDrawTarget::stderr(),
    ));
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("##-"),
    );

    // Process files in parallel
    mp3_files.par_iter().try_for_each(|file| -> Result<()> {
        let spectrogram = generate_spectrogram(file, args.fft_size, args.hop_size)?;
        let output_path = Path::new(&args.output_dir)
            .join(file.file_stem().unwrap())
            .with_extension(&args.output_format);
        save_spectrogram(&spectrogram, &output_path, &args.output_format)?;
        progress_bar.inc(1);
        Ok(())
    })?;

    Ok(())
}

fn generate_spectrogram(file_path: &Path, fft_size: usize, hop_size: usize) -> Result<Array2<f32>> {
    // Open the media source
    let src = std::fs::File::open(file_path)
        .with_context(|| format!("Failed to open file: {}", file_path.display()))?;
    let mss = MediaSourceStream::new(Box::new(src), Default::default());

    // Create a hint to help the format registry guess what format reader is appropriate
    let hint = Hint::new();

    // Use the default options for metadata and format readers
    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();

    // Probe the media source
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &fmt_opts, &meta_opts)
        .context("Failed to probe media source")?;

    // Get the format reader
    let mut format = probed.format;

    // Find the first audio track
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec == CODEC_TYPE_MP3)
        .context("No MP3 track found")?;

    // Create a decoder for the track
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("Failed to create decoder")?;

    // Set up the FFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);

    let mut samples_buffer = Vec::new();

    // Decode and process audio data
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(_) => break,
        };

        let decoded = decoder.decode(&packet).context("Failed to decode packet")?;
        let mut sample_buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
        sample_buf.copy_interleaved_ref(decoded);

        samples_buffer.extend_from_slice(sample_buf.samples());
    }

    // Calculate the number of time steps
    let time_steps = (samples_buffer.len() - fft_size) / hop_size + 1;

    // Create the spectrogram using from_shape_fn
    let spectrogram = Array2::from_shape_fn((fft_size / 2, time_steps), |(freq_bin, time_step)| {
        let start = time_step * hop_size;
        let mut buffer: Vec<Complex<f32>> = samples_buffer[start..start + fft_size]
            .iter()
            .map(|&s| Complex::new(s, 0.0))
            .collect();

        fft.process(&mut buffer);

        let magnitude = buffer[freq_bin].norm();
        let epsilon = 1e-10; // Small value to prevent log(0)
        (magnitude + epsilon).abs().ln()
    });

    Ok(spectrogram)
}

fn save_spectrogram(spectrogram: &Array2<f32>, output_path: &Path, format: &str) -> Result<()> {
    let spectrogram = spectrogram.t();
    let (width, height) = spectrogram.dim();

    // Create viridis colormap
    let colormap = colorgrad::viridis();

    // Find min and max values for normalization
    let min_value = spectrogram.iter().fold(f32::MAX, |a, &b| a.min(b));
    let max_value = spectrogram.iter().fold(f32::MIN, |a, &b| a.max(b));

    // Create image buffer
    let mut img = ImageBuffer::new(width as u32, height as u32);

    // Fill image buffer with spectrogram data
    for (x, row) in spectrogram.rows().into_iter().enumerate() {
        for (y, &value) in row.iter().enumerate() {
            let normalized_value = if max_value > min_value {
                (value - min_value) / (max_value - min_value)
            } else {
                0.5 // Default to middle value if max == min
            };

            // Ensure normalized_value is within [0, 1]
            let clamped_value = normalized_value.max(0.0).min(1.0);

            let color = colormap.at(clamped_value as f64).to_rgba8();
            img.put_pixel(
                x as u32,
                height as u32 - y as u32 - 1,
                Rgb([color[0], color[1], color[2]]),
            );
        }
    }

    // Save image
    match format.to_lowercase().as_str() {
        "png" => img
            .save_with_format(output_path, image::ImageFormat::Png)
            .with_context(|| {
                format!(
                    "Failed to save PNG spectrogram image: {}",
                    output_path.display()
                )
            })?,
        "jpeg" | "jpg" => img
            .save_with_format(output_path, image::ImageFormat::Jpeg)
            .with_context(|| {
                format!(
                    "Failed to save JPEG spectrogram image: {}",
                    output_path.display()
                )
            })?,
        "bmp" => img
            .save_with_format(output_path, image::ImageFormat::Bmp)
            .with_context(|| {
                format!(
                    "Failed to save BMP spectrogram image: {}",
                    output_path.display()
                )
            })?,
        _ => return Err(anyhow::anyhow!("Unsupported image format: {}", format)),
    }

    Ok(())
}
