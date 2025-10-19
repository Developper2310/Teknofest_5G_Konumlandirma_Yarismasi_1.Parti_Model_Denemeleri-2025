import pandas as pd

def remove_empty_f_rows(input_file, output_file):
    
  
    df = pd.read_excel(input_file)

    df_filtered = df[df['NR_Scan_NR_ARFCN'].notna()]

    df_filtered.to_excel(output_file, index=False)

    print(f"İşlem tamamlandı. Yeni dosya kaydedildi: {output_file}")


input_excel = "SCANNER sfd.xlsx"  
output_excel = "SCANNER_ok.xlsx"  

remove_empty_f_rows(input_excel,output_excel)