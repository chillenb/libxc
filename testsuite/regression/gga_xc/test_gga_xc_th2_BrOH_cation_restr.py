
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_th2_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th2", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-4.691828243020281e+01, -4.691824149797383e+01, -4.691874601463311e+01, -4.691933911245718e+01, -4.691893151846239e+01, -4.691893151846239e+01, -3.569193473197918e+00, -3.569220218108576e+00, -3.569926263289754e+00, -3.569370973245883e+00, -3.569310710837180e+00, -3.569310710837180e+00, -7.259631731399009e-01, -7.258560230239343e-01, -7.261516371540302e-01, -7.308661678965926e-01, -7.290699802532670e-01, -7.290699802532670e-01, -2.329991127913243e-01, -2.339846500224204e-01, -8.337830817000667e-01, -2.020304062509725e-01, -2.122087583869739e-01, -2.122087583869738e-01, -1.293653108974431e-02, -1.246493331119899e-02, -9.280340972997585e-02, -7.041743100295529e-03, -3.762340460431490e-03, -3.762340460431430e-03, -5.466905016471584e+00, -5.469011206323314e+00, -5.467000775167682e+00, -5.468860230042244e+00, -5.467973976219654e+00, -5.467973976219654e+00, -2.088608627808763e+00, -2.098821964164387e+00, -2.087929635106095e+00, -2.096663815230321e+00, -2.094867549647216e+00, -2.094867549647216e+00, -6.173017998004545e-01, -6.522295671313123e-01, -5.765690890555525e-01, -5.813719436343553e-01, -6.243298690949862e-01, -6.243298690949862e-01, -1.696188341173009e-01, -2.572898441381807e-01, -1.601630358705338e-01, -2.005364702979488e+00, -1.746434695252313e-01, -1.746434695252313e-01, 2.823056227669055e-03, -2.427916258048601e-03, 3.298401471465050e-02, -1.184951429098721e-01, 2.068024244102780e-02, 2.068024244102776e-02, -6.015155915089606e-01, -6.008729068309779e-01, -6.009973083002198e-01, -6.011648717842503e-01, -6.010710407617411e-01, -6.010710407617411e-01, -5.848247508111966e-01, -5.411307702570282e-01, -5.511480839027376e-01, -5.624252171206404e-01, -5.563152821864408e-01, -5.563152821864408e-01, -6.835438196241896e-01, -3.022327172905613e-01, -3.362853442277760e-01, -3.948280170832669e-01, -3.623181001852306e-01, -3.623181001852305e-01, -5.016784497961276e-01, -9.365429928547338e-02, -1.118581399014025e-01, -3.725141432950054e-01, -1.320220870278873e-01, -1.320220870278872e-01, -2.672284921636188e-02, 3.281298969473989e-02, 2.070642005091552e-02, -1.271816849470309e-01, 2.503991690582255e-02, 2.503991690582260e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_th2_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th2", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-5.287710983828823e+01, -5.287507529881977e+01, -5.286988588890350e+01, -5.290007510447845e+01, -5.288370893935088e+01, -5.288370893935088e+01, -4.996564563391738e+00, -4.996634890566394e+00, -4.998396262005389e+00, -4.996397632843760e+00, -4.996784145983717e+00, -4.996784145983717e+00, -8.150968953194088e-01, -8.131196691957703e-01, -7.623356851786749e-01, -7.698496194552084e-01, -7.695015005199742e-01, -7.695015005199742e-01, -2.079745535800548e-01, -2.125970973276577e-01, -9.601468541986867e-01, -1.435366007242853e-01, -1.702478702273131e-01, -1.702478702273130e-01, 8.906752207576181e-02, 8.704850076609649e-02, 2.179098208262513e-02, 9.932651599699195e-02, 9.219180374614483e-02, 9.219180374614479e-02, -7.681368932096079e+00, -7.679864737085969e+00, -7.681318711428420e+00, -7.679990788422896e+00, -7.680598561360685e+00, -7.680598561360685e+00, -2.271724308404397e+00, -2.304168935487499e+00, -2.225464187095775e+00, -2.254352728313740e+00, -2.311791452631732e+00, -2.311791452631732e+00, -7.521763990274393e-01, -8.523745626950284e-01, -6.917602624145823e-01, -7.481122619770538e-01, -7.685408985941522e-01, -7.685408985941522e-01, -7.733734463340783e-02, -1.906492944313627e-01, -6.769865961198593e-02, -2.793631228403735e+00, -1.096083550220408e-01, -1.096083550220408e-01, 9.695689100591789e-02, 9.666067842945641e-02, 7.850250981063966e-02, -2.553856522649597e-02, 8.328736537119143e-02, 8.328736537119143e-02, -7.962082933861517e-01, -7.827884475124073e-01, -7.873043354391731e-01, -7.911730911496251e-01, -7.892191999917446e-01, -7.892191999917446e-01, -7.771233529825564e-01, -6.081921517111619e-01, -6.548733713212450e-01, -7.021075823679904e-01, -6.780831679606563e-01, -6.780831679606563e-01, -8.934848353085609e-01, -2.483929594452470e-01, -3.098919284158532e-01, -4.352857503037652e-01, -3.684200667536722e-01, -3.684200667536721e-01, -5.615229851542169e-01, 2.723734230747658e-02, 1.260961822981031e-03, -4.295803177961245e-01, -5.669132977189562e-02, -5.669132977189562e-02, 8.485164648828929e-02, 8.230075675055913e-02, 8.869636822453673e-02, -4.880651049683569e-02, 8.178800339991975e-02, 8.178800339991976e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_th2_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th2", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-8.315408973411613e-08, -8.316011746288192e-08, -8.317670783620247e-08, -8.308724705220537e-08, -8.313557048879446e-08, -8.313557048879446e-08, 3.995714780087830e-06, 3.996768892638603e-06, 4.021071826353294e-06, 3.977922151905522e-06, 3.996852338136481e-06, 3.996852338136481e-06, -5.008316470610013e-03, -5.038603471219321e-03, -5.797074142445278e-03, -5.602318391381153e-03, -5.637273346307233e-03, -5.637273346307233e-03, -7.968273683832773e-01, -7.640920248437111e-01, -2.586990250729107e-03, -1.864937844676172e+00, -1.315795858007278e+00, -1.315795858007278e+00, -2.137177787809327e+04, -1.845238427793194e+04, -1.077188667493479e+02, -1.075513188523952e+05, -5.582583618463376e+04, -5.582583618463376e+04, 1.465139035596968e-06, 1.460057754516378e-06, 1.464917858434227e-06, 1.460432671393184e-06, 1.462563719799750e-06, 1.462563719799750e-06, -5.472982686597673e-05, -5.194658165826514e-05, -5.816515070476159e-05, -5.561906299745477e-05, -5.154626515311440e-05, -5.154626515311440e-05, -8.245413848996394e-03, -4.130895173595175e-03, -1.167685556153717e-02, -8.976937552090362e-03, -7.590280705801328e-03, -7.590280705801328e-03, -5.701981969430426e+00, -7.265392241964612e-01, -7.300974471535481e+00, 1.272521057331731e-04, -3.598729630019859e+00, -3.598729630019859e+00, -2.363910890546395e+05, -1.164827310964824e+05, -3.033272150098491e+05, -2.532438713104230e+01, -1.628247574867595e+05, -1.628247574867595e+05, -4.619194436193953e-03, -6.840638901032854e-03, -6.354593146358785e-03, -5.787648199997760e-03, -6.096941039453622e-03, -6.096941039453622e-03, -3.921902961880312e-03, -1.766034075922698e-02, -1.459042094710559e-02, -1.183970464225870e-02, -1.322716582765338e-02, -1.322716582765338e-02, -3.172405132723119e-03, -3.317815558133577e-01, -1.801781044148373e-01, -7.023973871896766e-02, -1.136052960448643e-01, -1.136052960448644e-01, -2.464584904216079e-02, -1.218399912795970e+02, -4.783995936124578e+01, -8.505304496739326e-02, -1.235135926235828e+01, -1.235135926235828e+01, -7.560197513280056e+03, -6.459739808712614e+06, -6.871417094194730e+05, -1.495053906004622e+01, -2.091596707769868e+05, -2.091596707769874e+05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05