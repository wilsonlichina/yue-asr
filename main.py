import boto3
import json
import os
import csv
import pandas as pd


# 创建SageMaker运行时客户端
runtime_client = boto3.client('sagemaker-runtime', region_name='us-east-1')


# 准备音频数据
def prepare_audio_data(audio_file_path):
    with open(audio_file_path, 'rb') as audio_file:
        audio_data = audio_file.read()
    return audio_data


# 调用端点进行推理
def invoke_whisper_endpoint(endpoint_name, audio_data):
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='audio/wav',  # 根据您的端点配置可能需要调整
        Body=audio_data
    )
    
    # 解析响应
    result = json.loads(response['Body'].read().decode())
    return result


# 读取标签数据
def load_labels(csv_path):
    labels = {}
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        labels[row['id']] = row['label']
    return labels


# 将结果写入CSV文件
def write_results_to_csv(results, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['filename', 'transcription', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    print(f"结果已保存至 {output_file}")


def main():
    endpoint_name = 'endpoint-asr-whisper-lvt'
    test_dir = './test'
    labels_file = 'test_audio_label.csv'
    output_file = 'result-whisper.csv'
    
    # 加载标签数据
    labels = load_labels(labels_file)
    
    # 获取测试文件夹中所有WAV文件
    audio_files = [f for f in os.listdir(test_dir) if f.endswith('.wav')]
    
    results = []
    
    # 遍历处理每个音频文件
    for i, filename in enumerate(audio_files):
        file_path = os.path.join(test_dir, filename)
        full_path = f"test/{filename}"  # 用于匹配标签的路径
        
        print(f"处理 {i+1}/{len(audio_files)}: {filename}")
        
        try:
            # 准备音频数据
            audio_data = prepare_audio_data(file_path)
            
            # 调用端点获取转录结果
            result = invoke_whisper_endpoint(endpoint_name, audio_data)
            transcription = result.get('text', '')
            
            # 查找对应的标签
            label = labels.get(full_path, "未找到标签")
            
            # 添加到结果列表
            results.append({
                'filename': file_path,
                'transcription': transcription,
                'label': label
            })
            
            print(f"转录结果: {transcription}")
            print(f"标签: {label}")
            print("-" * 50)
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")
    
    # 将结果写入CSV文件
    write_results_to_csv(results, output_file)


if __name__ == "__main__":
    main()
