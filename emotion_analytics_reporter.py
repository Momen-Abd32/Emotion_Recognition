"""
Emotion Analytics Reporting Module
"""

import os
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from fpdf import FPDF
import json
import cv2

class EmotionAnalyticsReporter:
    """Class to generate emotion analysis reports"""

    def __init__(self, output_dir="reports"):
        """
        Initialize the reporter

        Parameters:
            output_dir (str): directory to save reports
        """
        self.output_dir = output_dir
        self.emotions_data = []
        self.session_start_time = time.time()
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        os.makedirs(output_dir, exist_ok=True)
        self.charts_dir = os.path.join(output_dir, "charts")
        os.makedirs(self.charts_dir, exist_ok=True)

    def add_emotion_data(self, emotion_data):
        """Add emotion data to the report"""
        emotion_data['timestamp'] = time.time()
        emotion_data['datetime'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.emotions_data.append(emotion_data)

    def save_data_to_csv(self):
        if not self.emotions_data:
            return None
        df = pd.DataFrame(self.emotions_data)
        csv_path = os.path.join(self.output_dir, f"emotions_data_{self.session_id}.csv")
        df.to_csv(csv_path, index=False)
        return csv_path

    def save_data_to_json(self):
        if not self.emotions_data:
            return None
        json_path = os.path.join(self.output_dir, f"emotions_data_{self.session_id}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.emotions_data, f, ensure_ascii=False, indent=4)
        return json_path

    def generate_emotion_distribution_chart(self):
        if not self.emotions_data:
            return None
        primary_emotions = [d.get('primary_emotion') for d in self.emotions_data if d.get('primary_emotion')]
        if not primary_emotions:
            return None

        counts = {}
        for e in primary_emotions:
            counts[e] = counts.get(e, 0) + 1

        plt.figure(figsize=(10, 8))
        plt.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize':14})
        plt.axis('equal')
        plt.title('Primary Emotion Distribution', fontsize=16)
        chart_path = os.path.join(self.charts_dir, f"emotion_distribution_{self.session_id}.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        return chart_path

    def generate_emotion_timeline_chart(self):
        if not self.emotions_data:
            return None
        data_with_time = [(d.get('datetime'), d.get('primary_emotion')) for d in self.emotions_data if d.get('primary_emotion') and d.get('datetime')]
        if not data_with_time:
            return None
        df = pd.DataFrame(data_with_time, columns=['datetime', 'emotion'])
        df['datetime'] = pd.to_datetime(df['datetime'])

        plt.figure(figsize=(12,6))
        unique_emotions = df['emotion'].unique()
        emotion_to_num = {e:i for i,e in enumerate(unique_emotions)}
        df['emotion_num'] = df['emotion'].map(emotion_to_num)

        for e in unique_emotions:
            mask = df['emotion'] == e
            plt.scatter(df[mask]['datetime'], df[mask]['emotion_num'], label=e, s=100)
        plt.plot(df['datetime'], df['emotion_num'], 'k--', alpha=0.3)
        plt.yticks(range(len(unique_emotions)), unique_emotions)
        plt.xlabel('Time', fontsize=14)
        plt.ylabel('Emotions', fontsize=14)
        plt.title('Emotion Timeline', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.xticks(rotation=45)
        chart_path = os.path.join(self.charts_dir, f"emotion_timeline_{self.session_id}.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        return chart_path

    def generate_dual_emotions_chart(self):
        if not self.emotions_data:
            return None
        dual_emotions = [d.get('dual_emotion') for d in self.emotions_data if d.get('dual_emotion')]
        if not dual_emotions:
            return None
        counts = {}
        for e in dual_emotions:
            counts[e] = counts.get(e, 0) + 1
        sorted_emotions = sorted(counts.items(), key=lambda x:x[1], reverse=True)
        emotions, values = zip(*sorted_emotions)

        plt.figure(figsize=(12,8))
        bars = plt.bar(emotions, values, color=sns.color_palette("husl", len(emotions)))
        for bar in bars:
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1, f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=12)
        plt.xlabel('Dual Emotions', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.title('Dual Emotion Distribution', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        chart_path = os.path.join(self.charts_dir, f"dual_emotions_{self.session_id}.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        return chart_path

    def generate_confidence_chart(self):
        if not self.emotions_data:
            return None
        data = [(d.get('primary_emotion'), d.get('primary_confidence')) for d in self.emotions_data if d.get('primary_emotion') and d.get('primary_confidence') is not None]
        if not data:
            return None
        df = pd.DataFrame(data, columns=['emotion','confidence'])
        avg_conf = df.groupby('emotion')['confidence'].mean().reset_index()
        avg_conf = avg_conf.sort_values('confidence', ascending=False)

        plt.figure(figsize=(12,8))
        bars = plt.bar(avg_conf['emotion'], avg_conf['confidence'], color=sns.color_palette("muted", len(avg_conf)))
        for bar in bars:
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height()+0.01, f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=12)
        plt.xlabel('Emotion', fontsize=14)
        plt.ylabel('Average Confidence', fontsize=14)
        plt.title('Average Confidence per Emotion', fontsize=16)
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        chart_path = os.path.join(self.charts_dir, f"confidence_by_emotion_{self.session_id}.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        return chart_path

    def generate_pdf_report(self, include_images=True, max_images=5):
        if not self.emotions_data:
            return None
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial','B',16)
        pdf.cell(0,10,'Emotion Analysis Report',0,1,'C')
        pdf.set_font('Arial','',12)
        pdf.cell(0,10,f'Report Date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',0,1)
        duration = time.time() - self.session_start_time
        h,m,s = divmod(duration, 3600)[0], divmod(divmod(duration, 3600)[1], 60)[0], int(duration%60)
        pdf.cell(0,10,f'Session ID: {self.session_id}',0,1)
        pdf.cell(0,10,f'Session Duration: {int(h):02}:{int(m):02}:{s:02}',0,1)
        pdf.cell(0,10,f'Readings Count: {len(self.emotions_data)}',0,1)
        pdf.ln(10)

        pdf.set_font('Arial','B',14)
        pdf.cell(0,10,'Emotion Summary',0,1)
        primary_emotions = [d.get('primary_emotion') for d in self.emotions_data if d.get('primary_emotion')]
        if primary_emotions:
            counts = {}
            for e in primary_emotions: counts[e] = counts.get(e,0)+1
            sorted_counts = sorted(counts.items(), key=lambda x:x[1], reverse=True)
            pdf.set_font('Arial','B',12)
            pdf.cell(95,10,'Emotion',1,0,'C')
            pdf.cell(95,10,'Count',1,1,'C')
            pdf.set_font('Arial','',12)
            for e,c in sorted_counts:
                pdf.cell(95,10,e,1,0)
                pdf.cell(95,10,str(c),1,1,'C')
        pdf.ln(10)

        pdf.set_font('Arial','B',14)
        pdf.cell(0,10,'Charts',0,1)
        charts = []
        for func, title in [(self.generate_emotion_distribution_chart,'Primary Emotion Distribution'),
                            (self.generate_emotion_timeline_chart,'Emotion Timeline'),
                            (self.generate_dual_emotions_chart,'Dual Emotion Distribution'),
                            (self.generate_confidence_chart,'Average Confidence per Emotion')]:
            path = func()
            if path: charts.append((title,path))
        for title,path in charts:
            pdf.add_page()
            pdf.set_font('Arial','B',14)
            pdf.cell(0,10,title,0,1,'C')
            pdf.image(path,x=10,y=30,w=190)
        
        if include_images:
            face_images = [d.get('face_image') for d in self.emotions_data if d.get('face_image')]
            if face_images:
                pdf.add_page()
                pdf.set_font('Arial','B',14)
                pdf.cell(0,10,'Analyzed Face Images',0,1,'C')
                num_images = min(len(face_images), max_images)
                step = len(face_images)//num_images if num_images>0 else 1
                selected_images = face_images[::step][:num_images]
                for i,img in enumerate(selected_images):
                    if isinstance(img,str) and os.path.exists(img): img_path = img
                    elif isinstance(img,np.ndarray):
                        img_path = os.path.join(self.charts_dir,f"face_{i}_{self.session_id}.jpg")
                        cv2.imwrite(img_path,img)
                    else: continue
                    if i%2==0 and i>0: pdf.ln(70)
                    x_pos = 10 if i%2==0 else 110
                    pdf.image(img_path,x=x_pos,y=pdf.get_y(),w=90,h=60)

        report_path = os.path.join(self.output_dir, f"emotion_report_{self.session_id}.pdf")
        pdf.output(report_path)
        return report_path

    def generate_all_reports(self):
        reports = {}
        csv_path = self.save_data_to_csv()
        if csv_path: reports['csv'] = csv_path
        json_path = self.save_data_to_json()
        if json_path: reports['json'] = json_path
        pdf_path = self.generate_pdf_report()
        if pdf_path: reports['pdf'] = pdf_path
        return reports

# Example usage
if __name__=="__main__":
    test_data = [
        {'primary_emotion':'Happy','primary_confidence':0.85,'dual_emotion':'Happy + Surprise','dual_confidence':0.75},
        {'primary_emotion':'Sad','primary_confidence':0.72,'dual_emotion':'Slight Sad','dual_confidence':0.68},
        {'primary_emotion':'Angry','primary_confidence':0.91,'dual_emotion':'Angry + Fear','dual_confidence':0.82},
        {'primary_emotion':'Neutral','primary_confidence':0.65,'dual_emotion':'Neutral','dual_confidence':0.65},
        {'primary_emotion':'Happy','primary_confidence':0.78,'dual_emotion':'Slight Happy','dual_confidence':0.70},
    ]
    reporter = EmotionAnalyticsReporter()
    for d in test_data: reporter.add_emotion_data(d)
    reports = reporter.generate_all_reports()
    for t,p in reports.items(): print(f"{t} report created: {p}")
