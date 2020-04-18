#Yohann's work
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

def set_plot_config():
	sns.set()
	sns.set_style("whitegrid")
	sns.set_context("poster")
	mpl.rcParams['figure.figsize'] = [8.0, 6.0]
	mpl.rcParams['figure.dpi'] = 80
	mpl.rcParams['savefig.dpi'] = 100
	mpl.rcParams['font.size'] = 10
	mpl.rcParams['axes.labelsize'] = 10
	mpl.rcParams['axes.titlesize'] = 17
	mpl.rcParams['ytick.labelsize'] = 10
	mpl.rcParams['xtick.labelsize'] = 10
	mpl.rcParams['legend.fontsize'] = 'large'
	mpl.rcParams['figure.titlesize'] = 'medium'


def plot_test_distrib(y_proba, y_test, save_path, title):
	try:
		sns.distplot(proba[y_test==0, 1], label='b')
		sns.distplot(proba[y_test==1, 1], label='s')
		plt.xlabel('classifier score')
		plt.ylabel('density')
		plt.title(title)
		plt.legend()
		plt.savefig(save_path)
		plt.clf()
	except Exception as e:
		print('[WARNING] Plot test distrib failed')
		print('[WARNING] ', str(e))


def plot_scores(scores, scores_std, save_path, title):
	xx = np.arange(len(scores))
	try:
		plt.errorbar(xx, scores, yerr=scores_std, fmt='o',
		capsize=20, capthick=2, label='scores')
		plt.xlabel('iter num')
		plt.ylabel('scores')
		plt.title(title)
		plt.legend()
		plt.savefig(save_path)
		plt.clf()
	except Exception as e:
		print('[WARNING] Plot scores failed')
		print('[WARNING] ', str(e))
