
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_zvpbesol_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_zvpbesol", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.169595064838959e-02, -6.169681455617273e-02, -6.169927198515555e-02, -6.168644826855633e-02, -6.169336391560874e-02, -6.169336391560874e-02, -5.082214354138719e-02, -5.082731601661272e-02, -5.094713325146954e-02, -5.073944805188300e-02, -5.082839375662507e-02, -5.082839375662507e-02, -3.435058374838284e-02, -3.411915645799583e-02, -2.841297134814231e-02, -2.869261464299170e-02, -2.884103182475404e-02, -2.884103182475404e-02, -9.486321175543331e-03, -1.022468056132577e-02, -3.716059846264418e-02, -3.427154190750887e-03, -5.677762998037844e-03, -5.677762998037839e-03, -1.137375951559954e-08, -1.519968913920825e-08, -1.698174504218704e-05, -8.126950884825096e-10, -2.989327498187191e-09, -2.989327498187191e-09, -6.824054366381209e-02, -6.844406661058013e-02, -6.824893426528150e-02, -6.842860330633513e-02, -6.834410161952489e-02, -6.834410161952489e-02, -2.850927323843540e-02, -2.903280051395445e-02, -2.734039501875790e-02, -2.779298854062401e-02, -2.938198855423653e-02, -2.938198855423653e-02, -4.445960470736950e-02, -5.868761593155279e-02, -4.188631567476080e-02, -5.447146053801746e-02, -4.615622476758414e-02, -4.615622476758414e-02, -7.473159526851666e-04, -5.271487299734831e-03, -5.872637269355369e-04, -7.717718866035961e-02, -1.887273917198017e-03, -1.887273917198017e-03, -3.176682355007706e-10, -8.604066967044164e-10, -1.535616490373900e-09, -1.558441552044475e-04, -1.712795572380428e-09, -1.712795572380428e-09, -6.150473798742070e-02, -5.749112641053076e-02, -5.885036852017291e-02, -6.001719849128225e-02, -5.942892531798776e-02, -5.942892531798776e-02, -6.213754750768930e-02, -3.338567726534014e-02, -4.031067239576480e-02, -4.826147386547380e-02, -4.415276464215225e-02, -4.415276464215225e-02, -5.888237841972589e-02, -8.882050309382132e-03, -1.435952815277418e-02, -2.900038172420016e-02, -2.111761019453746e-02, -2.111761019453746e-02, -3.249175144393378e-02, -1.279654527240085e-05, -4.864388436418243e-05, -3.343082242378179e-02, -4.994853911026841e-04, -4.994853911026807e-04, -4.419897173566684e-08, -8.045012247846242e-12, -1.345574593731508e-10, -3.831370366131400e-04, -1.503540257428250e-09, -1.503540259596654e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_zvpbesol_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_zvpbesol", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.393511425780235e-01, -1.393520693247692e-01, -1.393547189630379e-01, -1.393409616821585e-01, -1.393483790923588e-01, -1.393483790923588e-01, -1.189849549115453e-01, -1.189896715362761e-01, -1.190989507441885e-01, -1.189109788080514e-01, -1.189908693365078e-01, -1.189908693365078e-01, -8.484347569087639e-02, -8.466612921604334e-02, -7.924772377583075e-02, -7.964679474036050e-02, -7.978311808713512e-02, -7.978311808713512e-02, -3.715675976709724e-02, -3.902405372066816e-02, -8.921846126820730e-02, -1.691551814246706e-02, -2.548939536538431e-02, -2.548939536538428e-02, -7.422084571730411e-08, -9.913804513415368e-08, -1.072932832250536e-04, -5.326986026808191e-09, -1.955982474677419e-08, -1.955982474767639e-08, -1.367496950519871e-01, -1.368891068409073e-01, -1.367554760845813e-01, -1.368785597568620e-01, -1.368207175032869e-01, -1.368207175032869e-01, -8.815574589343517e-02, -8.903289112327649e-02, -8.619469498860034e-02, -8.698462362230187e-02, -8.958125343529406e-02, -8.958125343529406e-02, -8.493519543732145e-02, -8.145549563791321e-02, -8.289770884440663e-02, -7.945282428481110e-02, -8.519754670962126e-02, -8.519754670962126e-02, -4.280805780784045e-03, -2.436302981903188e-02, -3.407955703508752e-03, -1.167538112681925e-01, -1.002963259831014e-02, -1.002963259831014e-02, -2.085840911708844e-09, -5.640518978373238e-09, -1.008287751179169e-08, -9.494799665791502e-04, -1.123321127098524e-08, -1.123321126733258e-08, -7.484147523041201e-02, -7.865984419815381e-02, -7.748565535603298e-02, -7.638107077065978e-02, -7.694915679224938e-02, -7.694915679224938e-02, -7.261426241833006e-02, -7.878910054349185e-02, -8.150939920380698e-02, -8.127754376798631e-02, -8.181227900996607e-02, -8.181227900996607e-02, -8.341711568780340e-02, -3.649750297653335e-02, -5.031863433953811e-02, -7.001395998798453e-02, -6.178708797658360e-02, -6.178708797658363e-02, -7.676922402974205e-02, -8.105697077395153e-05, -3.033232792096521e-04, -7.019882504607509e-02, -2.918197968892896e-03, -2.918197968892909e-03, -2.874664343115387e-07, -5.312568148849692e-11, -8.853101699782986e-10, -2.265280042739658e-03, -9.865829402827701e-09, -9.865829405084672e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_zvpbesol_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_zvpbesol", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.962116184365218e-10, 1.962148408064365e-10, 1.962188014197211e-10, 1.961710826990806e-10, 1.961975562975036e-10, 1.961975562975036e-10, 1.080141894174071e-06, 1.080339685846180e-06, 1.084798190220946e-06, 1.076071516773370e-06, 1.080252484833486e-06, 1.080252484833486e-06, 1.751250606770245e-03, 1.742134154479526e-03, 1.499221578443030e-03, 1.461513207223158e-03, 1.485633470061396e-03, 1.485633470061396e-03, 2.021443070851976e-01, 2.087983132403994e-01, 9.396399879663179e-04, 1.816049072303344e-01, 2.129743799473643e-01, 2.129743799473640e-01, 8.955489672148506e-03, 1.055020864076452e-02, 5.323110118182956e-02, 3.165644289451577e-03, 6.615649567108126e-03, 6.615649566897534e-03, 2.723931363333471e-07, 2.740312795776248e-07, 2.724583808980189e-07, 2.739043923052118e-07, 2.732265817124752e-07, 2.732265817124752e-07, 6.898142762208963e-06, 6.848794096984531e-06, 6.634964397231543e-06, 6.594069209589408e-06, 7.001597955372626e-06, 7.001597955372626e-06, 5.083114876313911e-03, 5.627304816964833e-03, 6.698021233548749e-03, 8.750708722072644e-03, 5.013693472759081e-03, 5.013693472759081e-03, 1.181895240821956e-01, 8.611762770619907e-02, 1.220291745722113e-01, 4.971015631493200e-05, 2.097174970901187e-01, 2.097174970901187e-01, 3.092182669560158e-03, 3.891755225997239e-03, 3.893550766559876e-02, 1.247344067967830e-01, 1.694487721532250e-02, 1.694487721631782e-02, 8.963333412859129e-03, 8.074474167881707e-03, 8.364687753097295e-03, 8.622784836123856e-03, 8.491679869889105e-03, 8.491679869889105e-03, 1.042974682601409e-02, 7.514657494196442e-03, 8.073521817247529e-03, 8.820269501353756e-03, 8.438662372334909e-03, 8.438662372334909e-03, 4.515901966157134e-03, 5.774298383983045e-02, 4.668716778855017e-02, 3.284871674189812e-02, 4.103861766679159e-02, 4.103861766679163e-02, 1.076275513062131e-02, 4.361106055935184e-02, 6.605596085743576e-02, 4.825727513217190e-02, 2.093791607542391e-01, 2.093791607542402e-01, 1.111865175982533e-02, 4.220044824426313e-03, 5.309577212848728e-03, 1.936340036092341e-01, 2.110061274660950e-02, 2.110061273885819e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05