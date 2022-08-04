
def changeFrequency(freq):
  f = open('c_lib_fast/c_src/qrsdet.h', "w")
  
  f.write("\n");
  f.write("#include <stdint.h>\n");
  f.write("\n");
  f.write("//#define SKIP_INIT\n");
  f.write("\n");
  f.write("#define SAMPLE_RATE	{:d}	/* Sample rate in Hz. */\n".format(round(freq)));
  ms_per_sample = 1000.0/freq
  f.write("#define MS10	{:d}\n".format(round(10/ms_per_sample))); 
  f.write("#define MS25	{:d}\n".format(round(25/ms_per_sample))); 
  f.write("#define MS30	{:d}\n".format(round(30/ms_per_sample))); 
  f.write("#define MS50	{:d}\n".format(round(50/ms_per_sample))); 
  f.write("#define MS80	{:d}\n".format(round(80/ms_per_sample))); 
  f.write("#define MS95	{:d}\n".format(round(95/ms_per_sample))); 
  f.write("#define MS100	{:d}\n".format(round(100/ms_per_sample))); 
  f.write("#define MS125	{:d}\n".format(round(125/ms_per_sample))); 
  f.write("#define MS150	{:d}\n".format(round(150/ms_per_sample))); 
  f.write("#define MS160	{:d}\n".format(round(160/ms_per_sample))); 
  f.write("#define MS175	{:d}\n".format(round(175/ms_per_sample))); 
  f.write("#define MS195	{:d}\n".format(round(195/ms_per_sample))); 
  f.write("#define MS200	{:d}\n".format(round(200/ms_per_sample))); 
  f.write("#define MS220	{:d}\n".format(round(220/ms_per_sample))); 
  f.write("#define MS250	{:d}\n".format(round(250/ms_per_sample))); 
  f.write("#define MS300	{:d}\n".format(round(300/ms_per_sample))); 
  f.write("#define MS360	{:d}\n".format(round(360/ms_per_sample))); 
  f.write("#define MS450	{:d}\n".format(round(450/ms_per_sample))); 
  f.write("#define MS1000	{:d}\n".format(round(freq)));
  f.write("#define MS1500	{:d}\n".format(round(1500/ms_per_sample))); 
  f.write("#define DERIV_LENGTH	MS10\n");
  f.write("#define LPBUFFER_LGTH {:d}\n".format(max(round(50/ms_per_sample),2)));
  f.write("#define HPBUFFER_LGTH MS125\n");
  f.write("\n");
  f.write("#define WINDOW_WIDTH	{:d}\n".format(max(round(80/ms_per_sample),1)));
  f.write("#define	FILTER_DELAY ({:d}+PRE_BLANK)\n".format(round(92.5/ms_per_sample)))
  f.write("#define DER_DELAY	WINDOW_WIDTH + FILTER_DELAY + MS100\n");
  f.write("\n");
  f.write("#define ECG_BUFFER_LENGTH {:d}\n".format(round(2*freq)));
  f.write("\n");
  f.write("#define AMPL_NORM_SHIFT 12\n");
  f.write("#define AMPL_NORM_TARGET 5000\n");
  f.write("#define GAIN_SMOOTH_DECAY_LOG 5\n");
  f.write("\n");
  f.write("int QRSDet( int datum, int fdatum, int init );\n");
  f.write("int QRSFilter(int datum,int init,int16_t* datum_filt);\n");
  f.write("int init_QRSNorm(void);\n");
  f.write("int QRSNorm(int datum);\n");
  f.write("void QRSNorm_updateGain(int amplitude);\n");

  f.close()
  
  f = open('c_lib_fast/c_src/upsample.h', "w")
  
  class_freq = 200;
  ratio = freq/class_freq;
  n = 32;
  log2n = 5;
  m = round(ratio*32);
  while m%2==0:
    m = m//2
    n = n//2
    log2n = log2n-1
  
  f.write("\n");
  f.write("#define LOG2_N	{:d}	\n".format(log2n));
  f.write("#define N	{:d}	\n".format(n));
  f.write("#define M	{:d}	\n".format(m));
  f.write("\n");
  f.write("int16_t* upsample(int16_t sample, int* length);\n");

  f.close()