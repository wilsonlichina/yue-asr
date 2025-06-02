import boto3
import json
import os
import csv
import pandas as pd
import time
import uuid
from urllib.parse import urlparse
import requests

# 创建SageMaker运行时客户端和Transcribe客户端
runtime_client = boto3.client('sagemaker-runtime', region_name='us-east-1')
transcribe_client = boto3.client('transcribe', region_name='us-east-1')
s3_client = boto3.client('s3', region_name='us-east-1')


# 从S3获取音频文件
def get_audio_from_s3(bucket_name, object_key):
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        audio_data = response['Body'].read()
        return audio_data
    except Exception as e:
        print(f"从S3获取音频文件时出错: {str(e)}")
        return None


# 准备音频数据
def prepare_audio_data(audio_file_path):
    with open(audio_file_path, 'rb') as audio_file:
        audio_data = audio_file.read()
    return audio_data


# 调用Whisper端点进行推理
def invoke_whisper_endpoint(endpoint_name, audio_data):
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='audio/wav',  # 根据您的端点配置可能需要调整
        Body=audio_data
    )
    
    # 解析响应
    result = json.loads(response['Body'].read().decode())
    return result


# 异步调用Amazon Transcribe
def invoke_async_transcribe(bucket_name, object_key, language_code='zh-HK'):
    job_name = f"transcribe-job-{uuid.uuid4()}"
    s3_uri = f"s3://{bucket_name}/{object_key}"
    
    try:
        response = transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': s3_uri},
            MediaFormat='wav',
            LanguageCode=language_code
        )
        
        # 等待转录作业完成
        while True:
            status = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
            if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
                break
            print(f"转录作业 {job_name} 正在进行中...")
            time.sleep(5)
        
        if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
            transcript_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
            # 获取转录结果
            
            response = requests.get(transcript_uri)
            transcript_data = json.loads(response.text)
            transcription = transcript_data['results']['transcripts'][0]['transcript']
            return transcription
        else:
            print(f"转录作业失败: {status['TranscriptionJob'].get('FailureReason', '未知错误')}")
            return ""
    except Exception as e:
        print(f"调用Amazon Transcribe时出错: {str(e)}")
        return ""


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
        fieldnames = ['filename', 'whisper_transcription', 'transcribe_transcription', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    print(f"结果已保存至 {output_file}")


def main():
    endpoint_name = 'endpoint-asr-whisper-lvt'
    s3_bucket = 'yue-asr-audio-uploads'
    labels_file = 'test_audio_label.csv'
    output_file = 'result-vs-transcribe.csv'
    
    # 加载标签数据
    labels = load_labels(labels_file)
    
    # 获取S3存储桶中的所有WAV文件
    response = s3_client.list_objects_v2(Bucket=s3_bucket)
    audio_files = [item['Key'] for item in response.get('Contents', []) if item['Key'].endswith('.wav')]
    
    results = []
    
    # 遍历处理每个音频文件
    for i, filename in enumerate(audio_files):
        print(f"处理 {i+1}/{len(audio_files)}: {filename}")
        
        try:
            # 从S3获取音频数据
            audio_data = get_audio_from_s3(s3_bucket, filename)
            if audio_data is None:
                continue
            
            # 构建用于匹配标签的路径
            base_filename = os.path.basename(filename)
            label_path = f"test/{base_filename}"
            
            # 调用Whisper端点获取转录结果
            whisper_result = invoke_whisper_endpoint(endpoint_name, audio_data)
            whisper_transcription = whisper_result.get('text', '')
            
            # 调用Amazon Transcribe获取转录结果
            transcribe_transcription = invoke_async_transcribe(s3_bucket, filename)
            
            # 查找对应的标签
            label = labels.get(label_path, "未找到标签")
            
            # 添加到结果列表
            results.append({
                'filename': filename,
                'whisper_transcription': whisper_transcription,
                'transcribe_transcription': transcribe_transcription,
                'label': label
            })
            
            print(f"Whisper转录结果: {whisper_transcription}")
            print(f"Transcribe转录结果: {transcribe_transcription}")
            print(f"标签: {label}")
            print("-" * 50)
            
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")
    
    # 将结果写入CSV文件
    write_results_to_csv(results, output_file)


if __name__ == "__main__":
    main()
