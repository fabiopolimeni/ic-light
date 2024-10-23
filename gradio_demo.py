import gradio as gr
import db_examples
from iclight_fc import IcLightFC, BGSource

# Initialize IC-Light
ic_light = IcLightFC()

quick_prompts = [
    'sunshine from window',
    'neon light, city',
    'sunset over sea',
    'golden time',
    'sci-fi RGB glowing, cyberpunk',
    'natural lighting',
    'warm atmosphere, at home, bedroom',
    'magic lit',
    'evil, gothic, Yharnam',
    'light and shadow',
    'shadow from window',
    'soft studio lighting',
    'home atmosphere, cozy bedroom illumination',
    'neon, Wong Kar-wai, warm'
]
quick_prompts = [[x] for x in quick_prompts]

quick_subjects = [
    'beautiful woman, detailed face',
    'handsome man, detailed face',
]
quick_subjects = [[x] for x in quick_subjects]

block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## IC-Light (Relighting with Foreground Condition)")
    with gr.Row():
        with gr.Column():
            with gr.Row():
                input_fg = gr.Image(source='upload', type="numpy", label="Image", height=480)
                output_bg = gr.Image(type="numpy", label="Preprocessed Foreground", height=480)
            prompt = gr.Textbox(label="Prompt")
            bg_source = gr.Radio(choices=[e.value for e in BGSource],
                               value=BGSource.NONE.value,
                               label="Lighting Preference (Initial Latent)", type='value')
            example_quick_subjects = gr.Dataset(samples=quick_subjects, label='Subject Quick List', samples_per_page=1000, components=[prompt])
            example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Lighting Quick List', samples_per_page=1000, components=[prompt])
            relight_button = gr.Button(value="Relight")

            with gr.Group():
                with gr.Row():
                    num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                    seed = gr.Number(label="Seed", value=12345, precision=0)
                with gr.Row():
                    image_width = gr.Slider(label="Image Width", minimum=256, maximum=1024, value=512, step=64)
                    image_height = gr.Slider(label="Image Height", minimum=256, maximum=1024, value=640, step=64)

            with gr.Accordion("Advanced options", open=False):
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1)
                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=2, step=0.01)
                lowres_denoise = gr.Slider(label="Lowres Denoise (for initial latent)", minimum=0.1, maximum=1.0, value=0.9, step=0.01)
                highres_scale = gr.Slider(label="Highres Scale", minimum=1.0, maximum=3.0, value=1.5, step=0.01)
                highres_denoise = gr.Slider(label="Highres Denoise", minimum=0.1, maximum=1.0, value=0.5, step=0.01)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality')
                n_prompt = gr.Textbox(label="Negative Prompt", value='lowres, bad anatomy, bad hands, cropped, worst quality')
        with gr.Column():
            result_gallery = gr.Gallery(height=832, object_fit='contain', label='Outputs')
    with gr.Row():
        dummy_image_for_outputs = gr.Image(visible=False, label='Result')
        gr.Examples(
            fn=lambda *args: ([args[-1]], None),
            examples=db_examples.foreground_conditioned_examples,
            inputs=[
                input_fg, prompt, bg_source, image_width, image_height, seed, dummy_image_for_outputs
            ],
            outputs=[result_gallery, output_bg],
            run_on_click=True, examples_per_page=1024
        )
    
    ips = [input_fg, prompt, image_width, image_height, num_samples, seed, steps, 
           a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source]
    
    relight_button.click(fn=ic_light.process_relight, inputs=ips, outputs=[output_bg, result_gallery])
    
    example_quick_prompts.click(
        lambda x, y: ', '.join(y.split(', ')[:2] + [x[0]]), 
        inputs=[example_quick_prompts, prompt], 
        outputs=prompt, 
        show_progress=False, 
        queue=False
    )
    
    example_quick_subjects.click(
        lambda x: x[0], 
        inputs=example_quick_subjects, 
        outputs=prompt, 
        show_progress=False, 
        queue=False
    )

block.launch(server_name='0.0.0.0')
