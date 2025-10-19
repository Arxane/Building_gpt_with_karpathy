        extra = set(sd_hf.keys()) - set(sd.keys())
        print("Missing local keys:", missing)
        print("Extra HF keys:", extra)