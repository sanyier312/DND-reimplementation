import concurrent.futures


def label_directory(input_dir, output_dir, threshold=0.1):
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有需要处理的 .gml 文件
    files_to_process = []
    for fname in os.listdir(input_dir):
        if fname.endswith(".gml"):
            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, fname)
            files_to_process.append((in_path, out_path, threshold))

    # 使用 ThreadPoolExecutor 并发执行任务
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for in_path, out_path, thr in files_to_process:
            futures.append(executor.submit(label_graph_file, in_path, out_path, thr))

        # 可选：等待所有任务完成并检查异常
        for future, (in_path, out_path, _) in zip(futures, files_to_process):
            try:
                future.result()
                print(f"✔ 完成: {out_path}")
            except Exception as exc:
                print(f"❌ 处理失败 {in_path}: {exc}")