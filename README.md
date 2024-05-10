First of all, credit where credit is due - the starting point for this work was following repository implementing YOLOv5:
https://github.com/Okery/YOLOv5-PyTorch/tree/master
The included checkpoint is from ultralytics repo:
https://github.com/ultralytics/yolov5


(Almost) entirety of work is confined to profiling.ipynb file, which should be fairly simple to go through (you can simply run all cells in order for default processing). Additionally for more comfortable log generation, script with same hardcoded values as the notebook can be ran (e.g. python3 run_profiling.py > results.log).

Steps to run:
1) pip install -r requirements.txt
2) python run_profiling.py
3) (optional) Browse (and edit/run) code in profiling.ipynb

My goals when writing this code were:
1) make reasonably large part of it universal and applicable to other models
2) limit changes to existing model code to absolute minimum

With that we have a universal "TimedModule" torch module which can wrap any torch module, with a replacer crawler that allows it to work on an already loaded model. This part is easy to modify to add additional behavior, change the way the results are displayed, etc - with very minor modifications necessary to achieve such goals. Additionally, selecting *which* exact module types we want to profile is very simple too - there may be cases where we're more interested in block-level performance, rather than individual operations. 
That being said this solution is not without faults - it completely omits non-module operations (e.g. sums for residual connections or multiplications for attention blocks). However, these *generally* take up negligible amount of time compared to rest of the network, and if one wanted to profile them individually as well, it might be a good idea to use a dedicated tool (or manually add profiling to every single operation in forward() of the model).

Pre- and post-processing profiling is extremely simple - though normally most of the pre-processing operations would be done inside the DataSet class, I figured I'll pull them out for individual profiling and lack of interference from multiple loading workers. Doesn't make much of a difference in this use-case, but for production deployment you generally want the pre- and post-processing to run asynchronously to maximize throughput - then the max processing frequency is dictated by the slowest of [preprocess, process, postprocess] as opposed to the sum of these 3.

Lastly - this implementation of YOLOv5 includes non-module processing of it's results (anchors -> bounding boxes) in forward method of its head component - to stick true to my goal 2), I have opted to merely add a profiling decorator to this function and functions in box_ops as some of them are used here, however delving deeper and wrapping its lowest-level elements in this way is most definitely doable if need be.

Other notes:
- the prints could be refined and return actual module name from model hierarchy instead of how they're handled right now - though the current log should be usable too as it lists the operations in execution order and includes their parameters info
- torch.cuda.synchronize() is necessary for profiling gpu operations, torch runs them asynchronously so without sync we will get deceptively low numbers
- many details were not specified in the instruction (e.g. output format, what should be drawn, whether output should be saved, format of printing the data) so I took my liberty picking these myself, mostly guided by convenience. Naturally it's not a problem to adjust these to a more concrete specification.
- the pipeline is most certainly not optimized - the most obvious part being running on fp32 - however switching that and checking profiling results with this setup would be simple (modifying preprocessing pipeline + setting model dtype to fp16)
- for actual serious benchmarking purposes, a mechanism of aggregating the results of individual runs should be added to ensure that our benchmark is not prone to outliers. Could be done via aggregating results from multiple text logs, putting them into some array instead of printing, modifying the wrapper component to collect multiple runs instead of printing, etc. - depends on the needs.
- i'm not sure what the purpose of including both png and jpg files was - unless you're looking to also benchmark decoding the image, in which case it's a matter of wrapping cv2.imread in dataset in a timer function. 
