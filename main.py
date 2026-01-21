from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
from manim import Group  # Ensure Group is imported
import random

class DefaultTemplate(VoiceoverScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        with self.voiceover(text="Hello and welcome. This is Leon.") as tracker:
            head = Circle(radius=0.5, color=WHITE).shift(UP * 1.5)
            body = Line(head.get_bottom(), head.get_bottom() + DOWN * 1.5, color=WHITE)
            arms = VGroup(
                Line(body.get_top() + DOWN * 0.5, body.get_top() + DOWN * 0.5 + LEFT, color=WHITE),
                Line(body.get_top() + DOWN * 0.5, body.get_top() + DOWN * 0.5 + RIGHT, color=WHITE)
            )
            legs = VGroup(
                Line(body.get_bottom(), body.get_bottom() + DOWN + LEFT * 0.5, color=WHITE),
                Line(body.get_bottom(), body.get_bottom() + DOWN + RIGHT * 0.5, color=WHITE)
            )

            # Group them together
            stick_figure = VGroup(head, body, arms, legs)
            stick_figure.scale(0.3)  # Scale the stick figure
            # stick_figure.shift(RIGHT * 3) # Move it to the right

            self.add(stick_figure)
            self.play(Create(stick_figure))
            self.wait(1)
        with self.voiceover(text="Leon is a psychology graduate and he is interested in learning more about AI.") as tracker:
            self.play(stick_figure.animate.shift(DOWN * 1.3 ).scale(3))
        
        with self.voiceover(text="He knows all about how our brain works, but doesn’t know anything about how artificial brains work.") as tracker:
        # self.wait(1)
            
            brain_img = ImageMobject("C:\\Users\\calvi\\Documents\\Projects\\ManimExperiment\\media\\images\\Brain.png")
            brain_img.scale(0.1)
            # brain_img.next_to(stick_figure,buff=0.5)
            brain_img.move_to(stick_figure.get_top()+DOWN*0.3)
            self.play(FadeIn(brain_img))

        with self.voiceover(text="So in this video, we will explore the fascinating world of artificial intelligence together with Leon.") as tracker:
            self.play(FadeOut(brain_img))
            self.play(stick_figure.animate.shift(UP * 1.3).scale(1/3))
            
            self.wait(1)
        
        self.play(FadeOut(stick_figure))
        self.wait(1)

class SecondScene(VoiceoverScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        MainTitle = Text("How does our brain work?").set(font_size=20, color=WHITE)
        with self.voiceover(text="We start by asking the question: how does our brain work?") as tracker:
            self.play(Write(MainTitle))
        self.play(FadeOut(MainTitle))
        neuron_img = ImageMobject("C:\\Users\\calvi\\Documents\\Projects\\ManimExperiment\\media\\images\\Neuron.png")
        neuron_img.scale(0.5)
        with self.voiceover(text="Our brain consist of many neurons. Neurons are the building blocks of our brain.") as tracker:            
            self.play(FadeIn(neuron_img))

        dashed_lines = Group()
        neurons = Group()
        connections = []         
        levels = 3

        with self.voiceover(text="""They communicate with each other through electrical and chemical signals,""") as tracker:
            pass

        with self.voiceover(text="""Neurons can be connected to many more neurons.""") as tracker:
            self.play(neuron_img.animate.scale(0.1).rotate(-0.5*PI).to_edge(LEFT*2.5))
            
            # --- Build Pyramid ---
            base_x = neuron_img.get_center()[0]
            base_y = neuron_img.get_center()[1]
            x_offset = 0.8
            y_offset = 0.8

            for col in range(levels):
                row_neurons = Group()
                for row in range(col + 1):
                    new_neuron = neuron_img.copy()
                    x = base_x + col * x_offset
                    y = base_y - (row - col / 2) * y_offset
                    new_neuron.move_to([x, y, 0])
                    row_neurons.add(new_neuron)
                neurons.add(row_neurons)

            self.play(FadeIn(neurons))
            # self.wait(1)

            # --- Create Connections ---
            for i in range(1, levels):
                prev_row = neurons[i - 1]
                curr_row = neurons[i]
                for p_idx, prev_neuron in enumerate(prev_row):
                    for c_idx, curr_neuron in enumerate(curr_row):
                        start = prev_neuron.get_center()
                        end = curr_neuron.get_center()
                        direction = end - start
                        direction_norm = direction / np.linalg.norm(direction)
                        
                        neuron_radius = 0.18
                        start_offset = start + direction_norm * neuron_radius
                        end_offset = end - direction_norm * neuron_radius
                        
                        line = DashedLine(
                            start_offset, end_offset,
                            dash_length=0.1, color=WHITE, stroke_width=1
                        )
                        dashed_lines.add(line)
                        
                        # NEW: Store the connection data explicitly
                        connections.append({
                            "line": line,
                            "start_neuron": prev_neuron,
                            "end_neuron": curr_neuron,
                            "start_pos": start_offset,
                            "end_pos": end_offset,
                            "layer_index": i - 1, # 0 = first gap, 1 = second gap
                            "usage_count": 0
                        })

            for conn in connections:
                 self.play(FadeIn(conn['line']), run_time=0.1)

        # --- Initial Signal Demo ---
        with self.voiceover(text="When a neuron is activated, it sends a signal to all the neurons it is connected to.") as tracker:
            # Simple demo: light up everything once
            for _ in range(2):
                for level_idx in range(levels - 1):
                    # Get all lines in this level
                    active_conns = [c for c in connections if c['layer_index'] == level_idx]
                    
                    # Create dots for this level
                    dots = [Circle(radius=0.03, color=YELLOW, fill_opacity=1).move_to(c['start_pos']) for c in active_conns]
                    
                    self.play(*[FadeIn(d) for d in dots], run_time=0.2)
                    self.play(*[d.animate.move_to(c['end_pos']) for d, c in zip(dots, active_conns)], run_time=0.5)
                    self.play(*[FadeOut(d) for d in dots], run_time=0.2)
                self.wait(1)
        # --- The Learning Loop (Repetitions + Thickness Logic) ---
        with self.voiceover(text="Whenever a connection is more frequently used, it becomes stronger over time. This is how the brain learns new things. AI Researchers have looked at this and tried to replicate this process in artificial neural networks.") as tracker:
            
            num_passes = 8
            
            for pass_num in range(num_passes):
                
                # 1. DECAY: Weaken ALL connections slightly (so unused ones get thinner)
                for conn in connections:
                    # Decrease current count by 0.5 (don't go below 0)
                    conn['usage_count'] = max(0, conn['usage_count'] - 0.2)

                # 2. LOGIC: Pick a random path through the network
                active_neurons = [neurons[0][0]] # Start at top
                signals_to_animate = [] 

                for level_idx in range(levels - 1):
                    next_active_neurons = []
                    for neuron in active_neurons:
                        # Find connections starting from this neuron
                        possible_outcomes = [c for c in connections if c['start_neuron'] == neuron and c['layer_index'] == level_idx]
                        
                        if possible_outcomes:
                            # Randomly pick 1 path to continue
                            # (Using k=1 ensures a clear single path, making "decisions" clearer)
                            chosen_conn = random.choice(possible_outcomes)
                            
                            # REINFORCE: Boost the chosen connection
                            chosen_conn['usage_count'] += 1.5 
                            
                            signals_to_animate.append(chosen_conn)
                            next_active_neurons.append(chosen_conn['end_neuron'])
                    
                    active_neurons = list(set(next_active_neurons))

                # 3. ANIMATION: Play the signals (The dots moving)
                for level_idx in range(levels - 1):
                    layer_signals = [s for s in signals_to_animate if s['layer_index'] == level_idx]
                    if layer_signals:
                        dots = [Circle(radius=0.04, color=YELLOW, fill_opacity=1).move_to(s['start_pos']) for s in layer_signals]
                        self.play(*[FadeIn(d) for d in dots], run_time=0.1)
                        self.play(*[d.animate.move_to(s['end_pos']) for d, s in zip(dots, layer_signals)], run_time=0.3)
                        self.play(*[FadeOut(d) for d in dots], run_time=0.1)

                # 4. THICKNESS UPDATE (MUST BE INDENTED INSIDE THE LOOP)
                style_anims = []
                for conn in connections:
                    # Calculate new width based on usage
                    # Base width 1.0, plus usage. Cap it at 8.0 max.
                    new_width = min(8.0, 1.0 + conn['usage_count'])
                    
                    # If line is very weak (near 0) and it's the final passes, fade it out
                    if new_width <= 1.0 and pass_num > 1:
                         style_anims.append(conn['line'].animate.set_opacity(0.2).set_stroke(width=1))
                    else:
                         style_anims.append(conn['line'].animate.set_opacity(1).set_stroke(width=new_width))
                
                # Play the thickness change immediately after the signals pass
                if style_anims:
                    self.play(*style_anims, run_time=0.5)
            
            self.wait(2)
        
        with self.voiceover(text="They have come up with something similar.") as tracker:
            self.play(FadeOut(neuron_img), FadeOut(neurons), FadeOut(dashed_lines))
        
        with self.voiceover(text="It consist of two phases. Phase one being the learning or training phase and Phase two being the inference phase. Inference in this case means running the trained model. In podcast about this is what used a lot.") as tracker:
            phase_one = Text("Phase 1: Learning / Training").set(font_size=20, color=WHITE)
            phase_two = Text("Phase 2: Inference").set(font_size=20, color=WHITE)
            phase_one.move_to(ORIGIN + UP * 0.2)
            phase_two.move_to(ORIGIN + DOWN * 0.2)
            self.play(Write(phase_one), Write(phase_two))
            self.wait(2)

class TrainingScene(VoiceoverScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        
        # --- (Existing Intro Part) ---
        mnist_img = ImageMobject("C:\\Users\\calvi\\Documents\\Projects\\ManimExperiment\\media\\images\\mnist_examples.png")
        mnist_img.scale(0.8).move_to(ORIGIN + DOWN * 0.1)

        with self.voiceover(text="In the learning phase, the model is trained on a large dataset. For this example we use a dataset containing handwritten numbers. During training we feed this images to the network and the model with slowly updates its parameters. ") as tracker:            
            self.play(FadeIn(mnist_img))
            title_text = Text("MNIST Dataset").set(font_size=20, color=WHITE).move_to(ORIGIN + UP * 0.9)
            self.play(Write(title_text, run_time=1))

        with self.voiceover(text="Leons asks: so what are these parameters?") as tracker:
            self.play(FadeOut(mnist_img), FadeOut(title_text))
            
            # Stick figure creation
            head = Circle(radius=0.5, color=WHITE).shift(UP * 1.5)
            body = Line(head.get_bottom(), head.get_bottom() + DOWN * 1.5, color=WHITE)
            arms = VGroup(
                Line(body.get_top() + DOWN * 0.5, body.get_top() + DOWN * 0.5 + LEFT, color=WHITE),
                Line(body.get_top() + DOWN * 0.5, body.get_top() + DOWN * 0.5 + RIGHT, color=WHITE)
            )
            legs = VGroup(
                Line(body.get_bottom(), body.get_bottom() + DOWN + LEFT * 0.5, color=WHITE),
                Line(body.get_bottom(), body.get_bottom() + DOWN + RIGHT * 0.5, color=WHITE)
            )
            stick_figure = VGroup(head, body, arms, legs).scale(0.3).move_to(ORIGIN + DOWN * 0.3)
            
            self.play(FadeIn(stick_figure))
            leons_text = Text("What are these parameters?").set(font_size=15, color=WHITE).move_to(ORIGIN + UP * 0.8 )
            self.play(Write(leons_text, run_time=1))
        
        # --- NEW ANIMATION STARTS HERE ---

        # 1. Setup the "Learned" Brain
        # We need to rebuild the neurons because it's a new scene, 
        # but we position them to the left to leave room for the math on the right.
        neuron_img = ImageMobject("C:\\Users\\calvi\\Documents\\Projects\\ManimExperiment\\media\\images\\Neuron.png")
        # neuron_img.neuron_img.scale(0.5)
        neurons = Group()
        connections = []
        levels = 3
        
        # Reuse pyramid logic, but shifted LEFT
        base_x = neuron_img.get_center()[0] - 0.9
        base_y = neuron_img.get_center()[1]
        x_offset = 0.8
        y_offset = 0.8

        for col in range(levels):
            row_neurons = Group()
            for row in range(col + 1):
                new_neuron = neuron_img.copy().scale(0.08).rotate(-0.5*PI)
                x = base_x + col * x_offset
                y = base_y - (row - col / 2) * y_offset
                new_neuron.move_to([x, y, 0])
                row_neurons.add(new_neuron)
            neurons.add(row_neurons)

        # Manually define a "Strong Path" (Indices of neurons in each layer)
        # Layer 0: Neuron 0 -> Layer 1: Neuron 0 -> Layer 2: Neuron 1
        strong_path_indices = [(0,0), (1,0), (2,1)] 

        # Create connections with pre-defined weights
        strong_lines = [] 
        for i in range(1, levels):
            prev_row = neurons[i - 1]
            curr_row = neurons[i]
            for p_idx, prev_neuron in enumerate(prev_row):
                for c_idx, curr_neuron in enumerate(curr_row):
                    start = prev_neuron.get_center()
                    end = curr_neuron.get_center()
                    direction = end - start
                    direction_norm = direction / np.linalg.norm(direction)
                    
                    # Offsets for the line drawing (so lines don't overlap image too much)
                    start_offset = start + direction_norm * 0.2
                    end_offset = end - direction_norm * 0.2
                    
                    # Logic: Is this part of our "Strong Path"?
                    is_strong = False
                    # Check if this connection links two nodes in our strong path list
                    if (i-1, p_idx) == strong_path_indices[i-1] and (i, c_idx) == strong_path_indices[i]:
                        is_strong = True

                    width = 6.0 if is_strong else 1.0
                    opacity = 1.0 if is_strong else 0.3
                    
                    line = DashedLine(
                        start_offset, end_offset,
                        dash_length=0.1, color=WHITE, stroke_width=width
                    ).set_opacity(opacity)
                    
                    conn_data = {
                        "line": line, 
                        "start_neuron": prev_neuron, # Store actual Mobject for center tracking
                        "end_neuron": curr_neuron,   # Store actual Mobject for center tracking
                        "start_pos": start,          # Use center for dots
                        "end_pos": end,              # Use center for dots
                        "is_strong": is_strong,
                        "layer_index": i-1
                    }
                    connections.append(conn_data)
                    if is_strong:
                        strong_lines.append(conn_data)

        brain_group = Group(neurons, *[c['line'] for c in connections])

        with self.voiceover(text="Good question Leon! Imagine again the brain that we had. With all the connections and neurons. This brain is from someone who has learned a lot and the connections are already built.") as tracker:
            self.play(FadeOut(stick_figure), FadeOut(leons_text))
            self.play(FadeIn(brain_group))
            self.wait(1)

        # --- Voiceover Part 2: Signal 0-100 ---
        with self.voiceover(text="A signal comes in. This for example could be a light signal, a sound, or a thought. For simplicity sake, lets imagine this signal as a value between 0 and 100, where 100 means the signal is very strong.") as tracker:
            # Start at the very first neuron (Top of pyramid)
            start_neuron = neurons[0][0]
            
            # Create initial signal at CENTER of neuron, size 0.05
            signal_dot = Circle(radius=0.05, color=YELLOW, fill_opacity=1).move_to(start_neuron.get_center())
            value_label = MathTex("Value = 0 - 100").move_to(UP*0.3 + LEFT).scale(0.2).set_color(WHITE)
            self.play(FadeIn(signal_dot), Write(value_label))
            self.wait(1)

        # --- UPDATED Voiceover Part 3: Complex Propagation ---
        with self.voiceover(text="Let's see how the signal originally would go through the network. It flows through every connection, but gets weaker on the thin paths and stays strong on the thick paths. It ends up at this neuron the strongest.") as tracker:
            self.play(FadeOut(value_label))
            # We track "active items" which are pairs of (Current Neuron, The Dot currently sitting there)
            active_items = [{'neuron': neurons[0][0], 'dot': signal_dot}]
            
            # Loop through the layers (0 and 1)
            for i in range(levels - 1):
                animations = []
                next_items = []
                dots_to_add = [] # Temporary list to add to scene
                
                # For every dot currently existing...
                for item in active_items:
                    current_neuron = item['neuron']
                    parent_dot = item['dot']
                    
                    # Find all connections leaving this neuron
                    outgoing_conns = [c for c in connections if c['start_neuron'] == current_neuron]
                    
                    for conn in outgoing_conns:
                        # 1. Spawn a new "child" dot at the center of the current neuron
                        # Start small (0.05)
                        child_dot = Circle(radius=0.05, color=YELLOW, fill_opacity=1).move_to(current_neuron.get_center())
                        dots_to_add.append(child_dot)
                        
                        # 2. Determine target scale
                        # If connection is strong, grow to 1.5x size (0.075 radius)
                        # If connection is weak, shrink to 0.4x size (0.02 radius)
                        target_scale = 1.5 if conn['is_strong'] else 0.4
                        
                        # 3. Create Animation
                        # Move to next center AND scale simultaneously
                        anim = child_dot.animate.move_to(conn['end_neuron'].get_center()).scale(target_scale)
                        animations.append(anim)
                        
                        # Record this dot for the next loop
                        next_items.append({'neuron': conn['end_neuron'], 'dot': child_dot})
                    
                    # Fade out the parent dot (it "split" into the children)
                    animations.append(FadeOut(parent_dot))

                # Add new dots to scene so they are visible
                self.add(*dots_to_add)
                
                # Play the movement/growth for this layer
                self.play(*animations, run_time=1.5)
                
                # Update loop list
                active_items = next_items

            # Move the label to the final strong dot
            # Find the strong dot in the final list
            final_strong_dot = next(item['dot'] for item in active_items if item['neuron'] == neurons[2][1]) # Index of final strong neuron
            
            self.wait(1)
            # Fade out all dots except final one for clarity
            other_dots = [item['dot'] for item in active_items if item['dot'] != final_strong_dot]
            self.play(*[FadeOut(d) for d in other_dots])
            self.wait(1)
            # Fade out all dots except final one for clarity
            other_dots = [item['dot'] for item in active_items if item['dot'] != final_strong_dot]
            self.play(*[FadeOut(d) for d in other_dots])
            
            # Increase size and make it empty with border only
            self.play(final_strong_dot.animate.scale(3).set_fill(opacity=0).set_stroke(width=3))

        with self.voiceover(text="How researchers simulate those connections is by using weights, basically multiplication factors. This connection has a higher weight than this one.") as tracker:
            self.wait(2)
            weights_text = Text("Weights = Multiplication Factors (W)").set(font_size=10, color=WHITE).move_to(LEFT+UP)
            self.play(Write(weights_text))
            # Wiggle a few connections to show they have weights
            example_conn = connections[0]  # Pick an example connection
            self.play(Wiggle(example_conn['line'], scale_value=1.3, rotation_angle=0.05*TAU), run_time=1)
            self.wait(0.5)
            example_conn = connections[1]  # Pick an example connection
            self.play(Wiggle(example_conn['line'], scale_value=1.3, rotation_angle=0.05*TAU), run_time=1)

        # with self.voiceover(text="So if we have our original input, let's say the value is 50. It get's multiplied by these factors and we end up with this.") as tracker:
        #     self.wait(2)
        #     weights_text2 = Text("Output = Input × W").set(font_size=10, color=WHITE).next_to(weights_text, DOWN*0.6, aligned_edge=LEFT)
        #     self.play(Write(weights_text2))
        #     self.wait(2)
        
        with self.voiceover(text="Let's see how this works in practice, by adding the weights to the first connections. Note that every connection has a weight, but we are only display two different ones right now.") as tracker:
            self.play(FadeOut(final_strong_dot), FadeOut(weights_text))
            weight_one = MathTex("2.0").set(font_size=8, color=WHITE).move_to(ORIGIN + LEFT * 0.5 + UP * 0.4)
            weight_two = MathTex("0.3").set(font_size=8, color=WHITE).move_to(ORIGIN + LEFT * 0.5 + DOWN * 0.4)
            self.play(Write(weight_one), Write(weight_two))
            self.wait(2)
        
        with self.voiceover(text="Let's also replace the neurons with circles, to make it cleaner.") as tracker:
            # Replace neurons with circles
            circle_neurons = Group()
            for neuron in neurons:
                for n in neuron:
                    circle = Circle(radius=0.15, color=WHITE, fill_opacity=0.0, stroke_width=0.5).move_to(n.get_center())
                    circle_neurons.add(circle)
                    print("neuron center:", n.get_center())
            self.play(FadeOut(neurons), FadeIn(circle_neurons))
            self.wait(2)

        with self.voiceover(text="Assume the input value is 50. That means that this neuron has a value of 50.") as tracker:
            
            input_label = MathTex("50").move_to(neurons[0][0].get_center()).scale(0.2).set_color(YELLOW)
            self.play(Write(input_label))
            self.wait(2)
        
        with self.voiceover(text="The signal then travels through the connections. On the strong connection, the value is multiplied by 2.0, resulting in 100 at the next neuron. On the weak connection, the value is multiplied by 0.3, resulting in 15 at its target neuron.") as tracker:
            
            # Animate signal on strong connection
            strong_conn = strong_lines[0]
            weak_conn = None
            for c in connections:
                if c['layer_index'] == 0 and not c['is_strong']:
                    weak_conn = c
                    break
            
            # Strong signal
            strong_dot = Circle(radius=0.05, color=YELLOW, fill_opacity=1).move_to(strong_conn['start_pos'])
            self.play(FadeIn(strong_dot))
            self.play(strong_dot.animate.move_to(strong_conn['end_pos']), run_time=1.5)
            strong_result = MathTex("100").move_to(strong_conn['end_neuron'].get_center()).scale(0.2).set_color(YELLOW)
            
            self.play(FadeOut(strong_dot))
            self.wait(1)
            self.play(Write(strong_result))
            # Weak signal
            weak_dot = Circle(radius=0.05, color=YELLOW, fill_opacity=1).move_to(weak_conn['start_pos'])
            self.play(FadeIn(weak_dot))
            self.play(weak_dot.animate.move_to(weak_conn['end_pos']), run_time=1.5)
            weak_result = MathTex("15").move_to(weak_conn['end_neuron'].get_center()).scale(0.2).set_color(YELLOW)
            
            self.play(FadeOut(weak_dot))
            self.wait(1)
            self.play(Write(weak_result))
            self.wait(2)
        
        with self.voiceover(text="Now the next layer is a bit more complex, because these have multiple inputs. Again, the value is getting multiplied by the weight. But this time its summed together.") as tracker:
            
            # 1. Define source values and targets
            print("Shape of circle_neurons:", len(circle_neurons))
            print("Neurons per layer:", [len(layer) for layer in neurons])
            l1_values = [100, 15] # The values currently sitting in Layer 1
            l1_nodes = circle_neurons[1:3]
            l2_nodes = circle_neurons[3:6]
            
            # 2. Animate Signals traveling to ALL nodes in Layer 2
            signal_anims = []
            
            # We need to calculate the math for the visual effects
            # Node 0 (Left):  100*0.3 + 15*0.3
            # Node 1 (Mid):   100*2.0 + 15*0.3  (The Strong Path)
            # Node 2 (Right): 100*0.3 + 15*0.3
            
            # Iterate through targets (Layer 2)
            for t_idx, target_node in enumerate(l2_nodes):
                
                # Iterate through sources (Layer 1)
                for s_idx, source_node in enumerate(l1_nodes):
                    # Determine start and end points
                    start_pos = source_node.get_center()
                    end_pos = target_node.get_center()
                    
                    # Determine if this specific link is strong or weak
                    # Based on your strong_path_indices: (1,0) -> (2,1) is the only strong one here
                    is_strong = (s_idx == 0 and t_idx == 1)
                    
                    # Create Dot
                    dot = Circle(radius=0.05, color=YELLOW, fill_opacity=1).move_to(start_pos)
                    
                    # Target scale (Big if strong, small if weak)
                    target_scale = 1.5 if is_strong else 0.5
                    
                    # Add animation (Create -> Move -> FadeOut)
                    # We use a Successversion of animations to keep the dot alive long enough to "hit"
                    signal_anims.append(
                        Succession(
                            FadeIn(dot, run_time=0.1),
                            dot.animate.move_to(end_pos).scale(target_scale).set_run_time(1.5),
                            FadeOut(dot, run_time=0.1)
                        )
                    )
            
            # Play all signal animations at once
            self.play(AnimationGroup(*signal_anims), run_time=2)

            # 3. Show the Math for the Center Node (The complex one)
            # Center node is l2_nodes[1]
            center_pos = l2_nodes[1].get_center()
            
            # Create the equation text: "2.0*100 + 0.3*15"
            # We position it slightly above the node first
            equation = MathTex("2.0 \\cdot 100", "+", "0.3 \\cdot 15").scale(0.3).next_to(l2_nodes[1], UP)
            equation[0].set_color(GREEN) # Strong part
            equation[2].set_color(WHITE) # Weak part
            
            self.play(Write(equation))
            self.wait(1)
            
            # Transform to intermediate numbers: "200 + 4.5"
            intermediate = MathTex("200", "+", "4.5").scale(0.3).next_to(l2_nodes[1], UP)
            intermediate[0].set_color(GREEN)
            
            self.play(Transform(equation, intermediate))
            self.wait(0.5)
            
            # 4. Final Summation for all nodes
            # Calculate final values
            # Left:  30 + 4.5 = 34.5
            # Mid:  200 + 4.5 = 204.5
            # Right: 30 + 4.5 = 34.5
            
            res_left = MathTex("34.5").scale(0.2).move_to(l2_nodes[0].get_center()).set_color(YELLOW)
            res_mid  = MathTex("204.5").scale(0.2).move_to(l2_nodes[1].get_center()).set_color(YELLOW)
            res_right= MathTex("34.5").scale(0.2).move_to(l2_nodes[2].get_center()).set_color(YELLOW)
            
            # Transform the equation into the middle result
            self.play(
                Transform(equation, res_mid),
                FadeIn(res_left),
                FadeIn(res_right)
            )
            
            self.wait(3)
        # --- Voiceover Part 4: Weights Explanation (The Math) ---
        # with self.voiceover(text="How researchers simulate those connections is by using weights, basically multiplication factors. This connection has a higher weight than this one.") as tracker:
            
        #     # Highlight the first strong connection
        #     focus_conn = strong_lines[0]
            
        #     # Dim everything else slightly to focus
        #     self.play(brain_group.animate.set_opacity(0.2), focus_conn['line'].animate.set_opacity(1))
            
        #     # Show "Input: 50" at start
        #     input_label = MathTex("Input: 50").next_to(focus_conn['start_pos'], UP).scale(0.6).set_color(YELLOW)
        #     self.play(Write(input_label))
            
        #     # Wiggle the line to show it is the "Weight"
        #     self.play(Wiggle(focus_conn['line'], scale_value=1.3, rotation_angle=0.05*TAU), run_time=1.5)
            
        #     # Label the weight
        #     weight_label = MathTex("Weight: 2.0").next_to(focus_conn['line'], UP).shift(DOWN*0.3).scale(0.6)
        #     self.play(Transform(input_label, weight_label))

        # with self.voiceover(text="So if we have our original input, let's say the value is 50. It get's multiplied by these factors and we end up with this.") as tracker:
            
        #     # Show the calculation clearly on the right side of the screen
        #     math_eq = MathTex("50", "\\times", "2.0", "=", "100")
        #     math_eq.scale(1.2).to_edge(RIGHT, buff=2)
            
        #     # Color coding
        #     math_eq[0].set_color(YELLOW) # Input
        #     math_eq[2].set_color(BLUE)   # Weight
            
        #     self.play(Write(math_eq))
        #     self.wait(1)
            
        #     # Show result appearing at the end of the line
        #     result_dot = Circle(radius=0.2, color=YELLOW, fill_opacity=1).move_to(focus_conn['end_pos'])
        #     result_label = MathTex("100").next_to(result_dot, RIGHT).scale(0.7)
            
        #     self.play(FadeIn(result_dot), Write(result_label))
        #     self.wait(1)
            
        #     # Cleanup math for next part
        #     self.play(FadeOut(input_label), FadeOut(weight_label), FadeOut(math_eq))


        # # --- Voiceover Part 5: Summation ---
        # with self.voiceover(text="Now the next layer is a bit more complex, because these have multiple inputs. Again, the value is getting multiplied by the weight. But this time its summed together.") as tracker:
            
        #     # Reset opacity
        #     self.play(brain_group.animate.set_opacity(1))
            
        #     # Focus on a neuron in Layer 2 (index 1) which has two incoming connections in our grid
        #     # Let's pick the one we just landed on (strong_lines[0]['end_neuron'])
        #     target_neuron_loc = strong_lines[0]['end_pos']
            
        #     # Find a second weak connection going to the SAME location
        #     weak_conn = None
        #     for c in connections:
        #         # Close enough match to target
        #         if np.linalg.norm(c['end_pos'] - target_neuron_loc) < 0.1 and c != focus_conn:
        #             weak_conn = c
        #             break
            
        #     if weak_conn:
        #         # Animate two signals arriving
        #         s1 = Circle(radius=0.15, color=YELLOW, fill_opacity=1).move_to(focus_conn['start_pos'])
        #         s2 = Circle(radius=0.08, color=YELLOW, fill_opacity=0.5).move_to(weak_conn['start_pos']) # Weak signal
                
        #         self.play(FadeIn(s1), FadeIn(s2))
                
        #         # Move them to the target
        #         self.play(
        #             s1.animate.move_to(focus_conn['end_pos']),
        #             s2.animate.move_to(weak_conn['end_pos']),
        #             run_time=1.5
        #         )
                
        #         # Show Summation Math
        #         # 100 (from strong) + 10 (from weak)
        #         sum_eq = MathTex("100", "+", "10", "=", "110")
        #         sum_eq.scale(1.2).to_edge(RIGHT, buff=2)
                
        #         self.play(Write(sum_eq))
                
        #         # Merge dots into one big dot
        #         final_dot = Circle(radius=0.25, color=GREEN, fill_opacity=1).move_to(target_neuron_loc)
        #         self.play(Transform(s1, final_dot), FadeOut(s2), FadeOut(result_dot), FadeOut(result_label))
                
        #         self.wait(2)