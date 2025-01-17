# Tracking results and evaluate them

📁 results
- 📁 data
    - 📁 gt : ground truth of datasets/benchmarks
        - 📁 UAVDT 
            - 📁 seqmaps - name of splits in the dataset
            - 📁 UAVDT-train - videos
                - 📁 video_1: 📄 gt/gt.txt,  📄 seqinfo.ini
                - ...
            - 📁 UAVDT-val
        - 📁 VisDrone
        - ...
    - 📁 trackers : results trackers
        - 📁 UAVDT
            - 📁 UAVDT-val
                - 📁 name_tracker_1
                    - 📄 data/video_1.txt: `frame_id,track_id,x1,y1,w,h,confidence,-1,-1,-1` ...
                    - summarry.txt, .csv, .png: Eval results trackers
                - ...
            - 📁 UAVDT-train
        - 📁 VisDrone
        - ....

