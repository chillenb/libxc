
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_8_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_8", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.358001920027839e+01, -2.358008266676406e+01, -2.358047147262554e+01, -2.357952958653093e+01, -2.358000789067404e+01, -2.358000789067404e+01, -3.385075271435787e+00, -3.385183982705405e+00, -3.388323325570036e+00, -3.389416016700251e+00, -3.388083116646435e+00, -3.388083116646435e+00, -5.972873524724377e-01, -5.966531567962706e-01, -5.827273898925809e-01, -5.934356585449987e-01, -5.903509614532613e-01, -5.903509614532613e-01, -1.710476893351144e-01, -1.737907153524786e-01, -6.546448178552936e-01, -1.204111742271874e-01, -1.383349299969493e-01, -1.383349299969492e-01, -5.503853269325552e-03, -5.787940560616908e-03, -3.163170969367920e-02, -3.138649740933540e-03, -3.964283299647313e-03, -3.964283299647312e-03, -5.752437256045611e+00, -5.752801653444282e+00, -5.752508186257672e+00, -5.752827703143316e+00, -5.752594585697715e+00, -5.752594585697715e+00, -2.066231778208006e+00, -2.095074143049057e+00, -2.058939827566616e+00, -2.085208839771705e+00, -2.085682667737989e+00, -2.085682667737989e+00, -6.134416576670024e-01, -6.601243248584340e-01, -5.413802620248241e-01, -5.534148313446396e-01, -6.259969165135570e-01, -6.259969165135570e-01, -8.277552689259414e-02, -1.657283913251780e-01, -7.637822607384802e-02, -1.895052417246532e+00, -9.884275076907797e-02, -9.884275076907796e-02, -2.356616739596776e-03, -3.026099965174863e-03, -2.341977878598387e-03, -5.116204304221491e-02, -2.794905456783322e-03, -2.794905456783322e-03, -6.354800740119880e-01, -6.362558576333427e-01, -6.360555410126545e-01, -6.358241874572151e-01, -6.359437486147151e-01, -6.359437486147151e-01, -6.130647124641908e-01, -5.417636752221963e-01, -5.675196455968124e-01, -5.882570112718964e-01, -5.777693654088628e-01, -5.777693654088628e-01, -6.824811968699365e-01, -2.161031733597401e-01, -2.634874753360560e-01, -3.492231488579731e-01, -3.071562170869311e-01, -3.071562170869311e-01, -4.758710762285457e-01, -3.047476550967491e-02, -4.121019364425205e-02, -3.429651242972439e-01, -6.568966079298487e-02, -6.568966079298488e-02, -7.556878533849662e-03, -7.801164920028892e-04, -1.731795214284961e-03, -6.205973705521541e-02, -2.601020181940381e-03, -2.601020181940378e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_8_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_8", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.875190872707860e+01, -2.875201892468194e+01, -2.875241959428157e+01, -2.875077831601968e+01, -2.875165103009364e+01, -2.875165103009364e+01, -4.714513884194955e+00, -4.714628144016096e+00, -4.717656737124654e+00, -4.716396226305736e+00, -4.716238352925383e+00, -4.716238352925383e+00, -8.192788565274844e-01, -8.177593618094209e-01, -7.839876031360566e-01, -8.000449649793128e-01, -7.960261165024620e-01, -7.960261165024620e-01, -2.129890602847517e-01, -2.170400076656003e-01, -8.711629936976454e-01, -1.499409281603190e-01, -1.724174667054049e-01, -1.724174667054050e-01, -6.631003062595942e-03, -6.981583088159267e-03, -3.873058902086084e-02, -3.845029702690430e-03, -4.823814737612782e-03, -4.823814737612781e-03, -7.079637946200797e+00, -7.083346230904509e+00, -7.079766121773471e+00, -7.083041463470470e+00, -7.081538137498982e+00, -7.081538137498982e+00, -2.580825083638544e+00, -2.601120300079308e+00, -2.563050595826303e+00, -2.581090124213359e+00, -2.600891813782678e+00, -2.600891813782678e+00, -8.047357200087278e-01, -9.089852924659774e-01, -7.416359611931248e-01, -8.009193925192708e-01, -8.201417423350498e-01, -8.201417423350498e-01, -1.021598253477713e-01, -2.045098153936838e-01, -9.434337694534781e-02, -2.748730188875912e+00, -1.215392243140752e-01, -1.215392243140751e-01, -2.989764904971286e-03, -3.774473072171570e-03, -2.875846145204180e-03, -6.312357598743004e-02, -3.474294805877388e-03, -3.474294805877388e-03, -8.418688314533277e-01, -8.234530559360012e-01, -8.297527678847554e-01, -8.351101245675906e-01, -8.324155498740304e-01, -8.324155498740304e-01, -8.247289339223089e-01, -6.495031450338450e-01, -6.870627285982001e-01, -7.348511811525852e-01, -7.094878641691397e-01, -7.094878641691398e-01, -9.537076610287123e-01, -2.682259355996736e-01, -3.319493621606527e-01, -4.600116585811799e-01, -3.920669024595488e-01, -3.920669024595487e-01, -6.057848666021818e-01, -3.700071022823578e-02, -5.063044076252120e-02, -4.521187195069309e-01, -8.076808427059208e-02, -8.076808427059201e-02, -9.438302322429997e-03, -1.017534843288445e-03, -2.105042280122531e-03, -7.576123493010881e-02, -3.209359515430585e-03, -3.209359515430579e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_8_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_8", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.617120325140113e-08, -1.617109913762520e-08, -1.617061089750370e-08, -1.617216885018053e-08, -1.617135601769139e-08, -1.617135601769139e-08, -1.386683271152490e-05, -1.386942759899673e-05, -1.393774805517474e-05, -1.391375141882092e-05, -1.391847135178364e-05, -1.391847135178364e-05, -4.666883577638306e-03, -4.654535223628589e-03, -4.377332518547270e-03, -4.657470918967062e-03, -4.582963990909503e-03, -4.582963990909503e-03, -7.236134629345413e-01, -7.267506703670313e-01, -1.129543339695743e-03, -5.918734215043376e-01, -6.548852413586999e-01, -6.548852413586996e-01, -2.791478595378764e+02, -2.559648314187685e+02, -5.686334761774760e+00, -6.623222643761138e+02, -5.117401443428883e+02, -5.117401443428878e+02, -4.796822736204474e-06, -4.796162255548860e-06, -4.796863717743461e-06, -4.796276733286188e-06, -4.796455088816931e-06, -4.796455088816931e-06, -1.303733524397418e-04, -1.315102503293535e-04, -1.301647208862017e-04, -1.313867536159936e-04, -1.311072934555034e-04, -1.311072934555034e-04, -2.333538703520130e-02, -1.926845582682046e-02, -2.593940985557731e-02, -2.569455081128027e-02, -2.275324062299688e-02, -2.275324062299688e-02, -8.930757754338710e-01, -3.749311048567093e-01, -1.029846562066332e+00, -2.027452420355794e-04, -9.946304651003178e-01, -9.946304651003182e-01, -7.419516039452290e+02, -5.891227118095063e+02, -3.131915736269454e+03, -2.354768084102907e+00, -1.248926599206904e+03, -1.248926599206904e+03, -2.999956737199261e-02, -3.041654879584397e-02, -3.028569070252712e-02, -3.016518125212837e-02, -3.022680881638665e-02, -3.022680881638665e-02, -3.311490294293269e-02, -4.548932736774492e-02, -4.351799350995108e-02, -4.008567278906628e-02, -4.196530300035804e-02, -4.196530300035804e-02, -1.522946369171819e-02, -2.606883913974587e-01, -1.976194224436823e-01, -1.222043297000069e-01, -1.691555790722618e-01, -1.691555790722619e-01, -5.339719476602407e-02, -6.487622974574034e+00, -3.185335440737964e+00, -1.769841409143216e-01, -1.869409074895190e+00, -1.869409074895191e+00, -8.045105628327414e+01, -5.933070780147688e+03, -4.026681606956770e+03, -2.250679564846574e+00, -1.816794414186208e+03, -1.816794414186213e+03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_8_BrOH_cation_restr_1_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_8", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_8_BrOH_cation_restr_1_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_8", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [2.168849036100676e-03, 2.168836118542966e-03, 2.168800744443762e-03, 2.168993185984296e-03, 2.168889210213085e-03, 2.168889210213085e-03, 8.025982880559781e-03, 8.027778743465301e-03, 8.077132037287611e-03, 8.077700496371895e-03, 8.069553668554840e-03, 8.069553668554840e-03, 1.467333060961686e-02, 1.460098556012632e-02, 1.321605644091212e-02, 1.533330551190807e-02, 1.468857723461434e-02, 1.468857723461434e-02, 7.542272024992110e-02, 7.857346029362930e-02, 2.399476056147939e-03, 2.054710225761883e-02, 3.586836181298721e-02, 3.586836181298719e-02, 9.636503380939014e-04, 1.023290157932582e-03, 3.525158490452995e-03, 3.956951832333238e-04, 6.349992641423551e-04, 6.349992641423528e-04, 8.579088846960771e-03, 8.567696172441266e-03, 8.578662753565620e-03, 8.568601496279967e-03, 8.573266915963449e-03, 8.573266915963449e-03, 1.832645016673189e-02, 1.895347919221576e-02, 1.830095349728738e-02, 1.890184050633353e-02, 1.868281450840251e-02, 1.868281450840251e-02, 6.854156592510499e-02, 6.069492995276066e-02, 5.928984483047256e-02, 5.725247454314891e-02, 6.866112892682807e-02, 6.866112892682807e-02, 9.992133116519501e-03, 3.687856811827234e-02, 8.946383320392812e-03, 1.746527230699146e-02, 2.007353840186567e-02, 2.007353840186566e-02, 1.475513816048378e-04, 2.849601964854439e-04, 7.685578834612487e-04, 6.044908850306801e-03, 4.865617908377901e-04, 4.865617908377886e-04, 6.470472439405975e-02, 6.890342854277427e-02, 6.745179706804245e-02, 6.622879188735339e-02, 6.684270190137225e-02, 6.684270190137229e-02, 6.572479647336775e-02, 9.472153999538445e-02, 9.161267801688316e-02, 8.281685944419012e-02, 8.766142601316147e-02, 8.766142601316139e-02, 5.610273142441138e-02, 5.547479361353930e-02, 7.060534850516405e-02, 8.372217701992431e-02, 8.652771622761365e-02, 8.652771622761347e-02, 8.419333437185667e-02, 3.729623067181479e-03, 4.321966574579672e-03, 1.043966907772246e-01, 1.058582190287041e-02, 1.058582190287040e-02, 6.007947436422991e-04, 2.737988699206066e-05, 4.183242028908078e-04, 1.109236514803989e-02, 5.956824294134740e-04, 5.956824294134673e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05