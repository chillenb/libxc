
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_q2d_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_q2d", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.177027855868470e-02, -5.177095698670406e-02, -5.177301630183482e-02, -5.176192737297024e-02, -5.177064442295388e-02, -5.177064442295388e-02, -4.138366104041656e-02, -4.138781633754447e-02, -4.150222168644322e-02, -4.130549413721850e-02, -4.138580210520840e-02, -4.138580210520840e-02, -2.588417766409537e-02, -2.566446707167936e-02, -2.002376960528476e-02, -2.031197453179221e-02, -2.580494609146811e-02, -2.580494609146811e-02, -8.879511945434638e-03, -9.563691932720615e-03, -3.285882438755888e-02, -2.374524379246585e-03, -9.090087004820063e-03, -9.090087004820063e-03, -6.375173343658198e-06, -5.965434165197664e-06, -3.347473999684369e-05, -2.716240614112816e-05, -5.759614943175072e-06, -5.759614943175072e-06, -5.862267646545734e-02, -5.882745164897550e-02, -5.864309683794256e-02, -5.880231927473414e-02, -5.872822252844592e-02, -5.872822252844592e-02, -2.048441338288406e-02, -2.093107454334527e-02, -1.986581319531926e-02, -2.020899156683736e-02, -2.161990663516267e-02, -2.161990663516267e-02, -3.896592613850931e-02, -5.635546723360698e-02, -3.753013859071292e-02, -5.102186061022936e-02, -4.144376463558565e-02, -4.144376463558565e-02, -6.147533563393295e-04, -4.240549055088748e-03, -6.770521962549127e-04, -7.183611871565769e-02, -1.348610457866016e-03, -1.348610457866016e-03, -2.638548933770682e-05, -2.235693907450905e-05, -8.519444999619655e-06, -1.349657736920146e-04, -1.539386682885767e-05, -1.539386682885090e-05, -6.149874150220740e-02, -5.637786725762337e-02, -5.812356801714558e-02, -5.954005749817527e-02, -5.882429047632420e-02, -5.882429047632420e-02, -6.169392790413110e-02, -2.895701351394838e-02, -3.627193172330719e-02, -4.454272237155665e-02, -4.022657778540525e-02, -4.022657778540525e-02, -5.636290826935742e-02, -7.292771829076890e-03, -1.205830924115500e-02, -2.440890164060311e-02, -1.769630588756011e-02, -1.769630588756011e-02, -2.731350553007566e-02, -2.180507082089223e-05, -7.295460206885698e-05, -2.927731582423964e-02, -4.278350690784615e-04, -4.278350690784615e-04, -7.445838711655360e-06, -2.866049959881082e-05, -2.838525764369131e-05, -4.333839020393987e-04, -1.175721207086698e-05, -1.175721207079031e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_q2d_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_q2d", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.220051871538122e-01, -1.220064667348186e-01, -1.220102531142154e-01, -1.219893367125550e-01, -1.220058780294000e-01, -1.220058780294000e-01, -1.090648301287216e-01, -1.090698283896362e-01, -1.092074440751401e-01, -1.089717434968915e-01, -1.090674302931856e-01, -1.090674302931856e-01, -7.647144823077927e-02, -7.617353873037050e-02, -6.713066787665900e-02, -6.770527630478829e-02, -7.636440839338166e-02, -7.636440839338166e-02, -3.580361806823747e-02, -3.764986095972048e-02, -8.620761519262196e-02, -1.236605551859022e-02, -3.638152360732680e-02, -3.638152360732680e-02, 2.953903839624984e-05, 2.723652275400090e-05, -1.880543211794936e-04, 1.142508709843015e-04, 2.625274059151219e-05, 2.625274059151219e-05, -1.293318042337075e-01, -1.295098095415515e-01, -1.293496008548832e-01, -1.294880158117208e-01, -1.294236453424433e-01, -1.294236453424433e-01, -7.276105772805830e-02, -7.376584681861806e-02, -7.135628718750540e-02, -7.214824180273754e-02, -7.526013490060514e-02, -7.526013490060514e-02, -8.420121286646440e-02, -8.277480212768665e-02, -8.264650051934563e-02, -8.231883044719883e-02, -8.556327870150461e-02, -8.556327870150461e-02, -3.556495759656908e-03, -2.052348423482998e-02, -3.897526296133448e-03, -1.178685654074653e-01, -7.418824851260258e-03, -7.418824851260258e-03, 1.106367886100694e-04, 9.888046620789700e-05, 3.974707217219212e-05, -8.141449057549111e-04, 7.001080069679187e-05, 7.001080069674549e-05, -7.565909723006001e-02, -8.031451455539318e-02, -7.891998273330404e-02, -7.764365716306494e-02, -7.830498074095187e-02, -7.830498074095187e-02, -7.371766927345201e-02, -7.630316064900296e-02, -8.123488191763154e-02, -8.281686427968807e-02, -8.247501606533626e-02, -8.247501606533625e-02, -8.474064481101361e-02, -3.183452675123221e-02, -4.564649727493618e-02, -6.709462978207249e-02, -5.746903911986784e-02, -5.746903911986786e-02, -7.338944013328760e-02, -1.054267996097481e-04, -4.341539797406441e-04, -6.916715051238850e-02, -2.514343897886454e-03, -2.514343897886452e-03, 3.401606395788470e-05, 1.387029398239796e-05, 9.231736065891284e-05, -2.545630974896713e-03, 5.378480696473599e-05, 5.378480696404227e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_q2d_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_q2d", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.756584567751193e-10, 1.756625219392935e-10, 1.756698789989750e-10, 1.756035321513009e-10, 1.756606906950014e-10, 1.756606906950014e-10, 1.078281855000123e-06, 1.078459223250847e-06, 1.083129374033972e-06, 1.073981312281474e-06, 1.078350541791280e-06, 1.078350541791280e-06, 1.699303660054709e-03, 1.686177888646520e-03, 1.331777915283774e-03, 1.308864649466232e-03, 1.694611367288956e-03, 1.694611367288956e-03, 2.070257062254547e-01, 2.156390856766667e-01, 1.011191213417748e-03, 1.494067355817849e-01, 2.099103107837121e-01, 2.099103107837121e-01, -1.968075983581167e+00, -1.588123847368028e+00, 5.965006231674979e-02, -5.909540683521071e+01, -1.610732370000810e+00, -1.610732370000810e+00, 2.836067632617852e-07, 2.854265525258920e-07, 2.837863818538102e-07, 2.852012750940560e-07, 2.845453871585410e-07, 2.845453871585410e-07, 6.043553275784024e-06, 6.032036073079703e-06, 5.850236747735771e-06, 5.843339652136294e-06, 6.293341182889959e-06, 6.293341182889959e-06, 6.263336713034181e-03, 7.960886129065736e-03, 7.605036149560293e-03, 1.036265894614063e-02, 5.767558646282456e-03, 5.767558646282456e-03, 9.348538983652689e-02, 6.850091556527776e-02, 1.152924376379503e-01, 6.082210115666546e-05, 1.458254055363187e-01, 1.458254055363187e-01, -6.570430105132850e+01, -3.887599093618462e+01, -6.200548210450692e+01, 1.040629513077055e-01, -4.393196645643045e+01, -4.393196645640776e+01, 1.206896424913576e-02, 1.045512500839479e-02, 1.098054577818271e-02, 1.142542690011051e-02, 1.119855018622047e-02, 1.119855018622047e-02, 1.419065632616146e-02, 7.942120760557300e-03, 9.065131307089469e-03, 1.046483883912042e-02, 9.732843741422696e-03, 9.732843741422701e-03, 6.343532224989745e-03, 4.862104977571693e-02, 4.306503662607732e-02, 3.556875502260261e-02, 4.082174759702752e-02, 4.082174759702755e-02, 1.202454781452579e-02, 3.566222361473205e-02, 6.793918181082186e-02, 6.088884738517177e-02, 1.553976596771499e-01, 1.553976596771508e-01, -1.225884528445396e+00, -1.038586412625559e+03, -3.411729054499689e+02, 2.010211850002376e-01, -6.634621330683413e+01, -6.634621330619963e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05