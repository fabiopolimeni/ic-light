import gradio as gr
from iclight_fc import IcLightFC, BGSource

ic_light = IcLightFC()

def process_image(input_fg, input_bg, prompt, width, height, samples, seed, steps, 
                 added_prompt, negative_prompt, cfg, highres_scale, highres_denoise, 
                 lowres_denoise, bg_source):
    
    if input_fg is None:
        raise gr.Error("Please provide a foreground image")
        
    bg_source_enum = BGSource[bg_source]
    
    _, result_images = ic_light.process_relight(
        input_fg=input_fg,
        input_bg=input_bg,
        prompt=prompt,
        image_width=width,
        image_height=height,
        num_samples=samples,
        seed=seed,
        steps=steps,
        a_prompt=added_prompt,
        n_prompt=negative_prompt,
        cfg=cfg,
        highres_scale=highres_scale,
        highres_denoise=highres_denoise,
        lowres_denoise=lowres_denoise,
        bg_source=bg_source_enum.value
    )
    
    return result_images

with gr.Blocks() as demo:
    gr.Markdown("# IC-Light Relighting Tool")
    
    with gr.Row():
        with gr.Column():
            input_fg = gr.Image(label="Foreground Image", type="numpy")
            input_bg = gr.Image(label="Background Image (Optional)", type="numpy", value=None)
            
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", placeholder="Enter your lighting prompt")
            width = gr.Slider(256, 1024, value=512, step=64, label="Width")
            height = gr.Slider(256, 1024, value=512, step=64, label="Height")
            samples = gr.Slider(1, 4, value=1, step=1, label="Number of Samples")
            seed = gr.Number(value=12345, label="Seed", precision=0)
            steps = gr.Slider(1, 50, value=10, step=1, label="Steps")
            
            with gr.Accordion("Advanced Settings", open=False):
                added_prompt = gr.Textbox(value="best quality", label="Additional Prompt")
                negative_prompt = gr.Textbox(
                    value="lowres, bad anatomy, bad hands, cropped, worst quality",
                    label="Negative Prompt"
                )
                cfg = gr.Slider(1.0, 10.0, value=2.0, step=0.1, label="CFG Scale")
                highres_scale = gr.Slider(1.0, 4.0, value=1.5, step=0.1, label="Highres Scale")
                highres_denoise = gr.Slider(0.1, 1.0, value=0.5, step=0.1, label="Highres Denoise")
                lowres_denoise = gr.Slider(0.1, 1.0, value=0.9, step=0.1, label="Lowres Denoise")
                bg_source = gr.Dropdown(
                    choices=[e.name for e in BGSource],
                    value=BGSource.NONE.name,
                    label="Background Source"
                )
    
    submit_btn = gr.Button("Generate")
    output_gallery = gr.Gallery(label="Results", columns=2, height=400)
    
    submit_btn.click(
        fn=process_image,
        inputs=[
            input_fg, input_bg, prompt, width, height, samples, seed, steps,
            added_prompt, negative_prompt, cfg, highres_scale, highres_denoise,
            lowres_denoise, bg_source
        ],
        outputs=output_gallery
    )

if __name__ == "__main__":
    demo.launch()
