#!/usr/bin/env python3
import argparse
from iclight_fc import IcLightFC, BGSource
import cv2
from datetime import datetime

def bg_source_type(value):
    return BGSource[value]

def main():
    parser = argparse.ArgumentParser(description='IC-Light Image Relighting Tool')
    parser.add_argument('-f', '--input_fg', required=True, help='Path to foreground image')
    parser.add_argument('-o', '--output', help='Path to output image (default: timestamp_mode.png)')
    parser.add_argument('-p', '--prompt', default='beautiful woman', help='Text prompt')
    parser.add_argument('-x', '--width', type=int, default=512, help='Output image width')
    parser.add_argument('-y', '--height', type=int, default=512, help='Output image height')
    parser.add_argument('-k', '--samples', type=int, default=1, help='Number of samples')
    parser.add_argument('-s', '--seed', type=int, default=12345, help='Random seed')
    parser.add_argument('-t', '--steps', type=int, default=25, help='Number of inference steps')
    parser.add_argument('-c', '--cfg', type=float, default=2.0, help='CFG scale')
    parser.add_argument('-l', '--lowres_denoise', type=float, default=0.9, help='Lowres denoise strength')
    parser.add_argument('-r', '--highres_scale', type=float, default=1.5, help='Highres scale')
    parser.add_argument('-d', '--highres_denoise', type=float, default=0.5, help='Highres denoise strength')
    parser.add_argument('-a', '--added_prompt', default='best quality', help='Additional positive prompt')
    parser.add_argument('-n', '--negative_prompt', default='lowres, bad anatomy, bad hands, cropped, worst quality', help='Negative prompt')
    parser.add_argument('-g', '--bg_source', type=bg_source_type, choices=list(BGSource), default=BGSource.NONE,
                help='Lighting preference: ' + ', '.join([e.name for e in BGSource]))

    args = parser.parse_args()

    # Generate output filename if not provided
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"{timestamp}_relight.png"

    # Initialize IC-Light
    ic_light = IcLightFC()

    # Load foreground image
    input_fg = cv2.imread(args.input_fg)
    input_fg = cv2.cvtColor(input_fg, cv2.COLOR_BGR2RGB)

    # Process the image
    _, results = ic_light.process_relight(
        input_fg=input_fg,
        prompt=args.prompt,
        image_width=args.width,
        image_height=args.height,
        num_samples=args.samples,
        seed=args.seed,
        steps=args.steps,
        a_prompt=args.added_prompt,
        n_prompt=args.negative_prompt,
        cfg=args.cfg,
        highres_scale=args.highres_scale,
        highres_denoise=args.highres_denoise,
        lowres_denoise=args.lowres_denoise,
        bg_source=args.bg_source.value
    )

    # Save the first result
    output = cv2.cvtColor(results[0], cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, output)
    print(f"Output saved to: {args.output}")

if __name__ == '__main__':
    main()
