import gradio as gr
import db_examples
from iclight import IcLight, BGSource

# Initialize IC-Light
ic_light = IcLight()

quick_prompts = [
    'beautiful woman',
    'handsome man',
    'beautiful woman, cinematic lighting',
    'handsome man, cinematic lighting',
    'beautiful woman, natural lighting',
    'handsome man, natural lighting',
    'beautiful woman, neo punk lighting, cyberpunk',
    'handsome man, neo punk lighting, cyberpunk',
]
quick_prompts = [[x] for x in quick_prompts]

block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## IC-Light (Relighting with Foreground and Background Condition)")
    with gr.Row():
        with gr.Column():
            with gr.Row():
                input_fg = gr.Image(source='upload', type="numpy", label="Foreground", height=480)
                input_bg = gr.Image(source='upload', type="numpy", label="Background", height=480)
            prompt = gr.Textbox(label="Prompt")
            bg_source = gr.Radio(choices=[e.value for e in BGSource],
                               value=BGSource.UPLOAD.value,
                               label="Background Source", type='value')

            example_prompts = gr.Dataset(samples=quick_prompts, label='Prompt Quick List', components=[prompt])
            bg_gallery = gr.Gallery(height=450, object_fit='contain', label='Background Quick List', value=db_examples.bg_samples, columns=5, allow_preview=False)
            relight_button = gr.Button(value="Relight")

            with gr.Group():
                with gr.Row():
                    num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                    seed = gr.Number(label="Seed", value=12345, precision=0)
                with gr.Row():
                    image_width = gr.Slider(label="Image Width", minimum=256, maximum=1024, value=512, step=64)
                    image_height = gr.Slider(label="Image Height", minimum=256, maximum=1024, value=640, step=64)

            with gr.Accordion("Advanced options", open=False):
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=7.0, step=0.01)
                highres_scale = gr.Slider(label="Highres Scale", minimum=1.0, maximum=3.0, value=1.5, step=0.01)
                highres_denoise = gr.Slider(label="Highres Denoise", minimum=0.1, maximum=0.9, value=0.5, step=0.01)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality')
                n_prompt = gr.Textbox(label="Negative Prompt",
                                    value='lowres, bad anatomy, bad hands, cropped, worst quality')
                normal_button = gr.Button(value="Compute Normal (4x Slower)")
        with gr.Column():
            result_gallery = gr.Gallery(height=832, object_fit='contain', label='Outputs')
    with gr.Row():
        dummy_image_for_outputs = gr.Image(visible=False, label='Result')
        gr.Examples(
            fn=lambda *args: [args[-1]],
            examples=db_examples.background_conditioned_examples,
            inputs=[
                input_fg, input_bg, prompt, bg_source, image_width, image_height, seed, dummy_image_for_outputs
            ],
            outputs=[result_gallery],
            run_on_click=True, examples_per_page=1024
        )

    ips = [input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, 
           a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source]
    
    relight_button.click(fn=ic_light.process_relight, inputs=ips, outputs=[result_gallery])
    normal_button.click(fn=ic_light.process_normal, inputs=ips, outputs=[result_gallery])
    example_prompts.click(lambda x: x[0], inputs=example_prompts, outputs=prompt, show_progress=False, queue=False)

    def bg_gallery_selected(gal, evt: gr.SelectData):
        return gal[evt.index]['name']

    bg_gallery.select(bg_gallery_selected, inputs=bg_gallery, outputs=input_bg)

block.launch(server_name='0.0.0.0')
