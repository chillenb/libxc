
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_tm_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_tm", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.226541057156514e+01, -2.226546415509095e+01, -2.226576824459874e+01, -2.226497151804720e+01, -2.226537958626370e+01, -2.226537958626370e+01, -3.402817609853170e+00, -3.402822782096344e+00, -3.403201930254911e+00, -3.405075296085304e+00, -3.403796868524643e+00, -3.403796868524643e+00, -6.699583671974413e-01, -6.697186112226842e-01, -6.646578272815148e-01, -6.684348515385246e-01, -6.671497927338912e-01, -6.671497927338912e-01, -2.012686423636713e-01, -2.027879605864270e-01, -8.277445894333907e-01, -1.606141583270465e-01, -1.753285090573230e-01, -1.753285090573231e-01, -2.020868588818759e-02, -2.070066046763453e-02, -6.299201994901746e-02, -1.425797946726783e-02, -1.613482328874790e-02, -1.613482328874790e-02, -5.413798412495116e+00, -5.414335536852542e+00, -5.413835617813069e+00, -5.414309310785527e+00, -5.414063814375026e+00, -5.414063814375026e+00, -2.105103829483879e+00, -2.121805964283363e+00, -2.103537209546571e+00, -2.118360610719563e+00, -2.115118579092785e+00, -2.115118579092785e+00, -5.907363613169941e-01, -6.103130259360690e-01, -5.356030870725844e-01, -5.339747125905379e-01, -5.992458907984598e-01, -5.992458907984598e-01, -1.239987153665441e-01, -2.101099464258919e-01, -1.164694507982986e-01, -1.808611294449750e+00, -1.377370411211530e-01, -1.377370411211530e-01, -1.226904162514063e-02, -1.377004307874502e-02, -9.794851305594638e-03, -8.590355169449101e-02, -1.176955190396499e-02, -1.176955190396500e-02, -5.887419172642298e-01, -6.014163151344002e-01, -5.999215813668393e-01, -5.970386409143041e-01, -5.987404618593465e-01, -5.987404618593465e-01, -5.468442238608270e-01, -5.241697327920189e-01, -5.406001255297060e-01, -5.566495242853602e-01, -5.482383271574004e-01, -5.482383271574004e-01, -6.368068675537123e-01, -2.569152597859692e-01, -2.954923804252881e-01, -3.590908601929507e-01, -3.267574163562766e-01, -3.267574163562766e-01, -4.740176948643919e-01, -6.234192311743284e-02, -7.559625606083670e-02, -3.412864098638981e-01, -1.011932243933372e-01, -1.011932243933372e-01, -2.478253370074983e-02, -6.941609507125469e-03, -9.045944266429837e-03, -9.756228227247249e-02, -1.098454136553187e-02, -1.098454136553186e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_tm_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_tm", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.793015269973021e+01, -2.793028155588111e+01, -2.793107355530835e+01, -2.792916721110073e+01, -2.793013467947106e+01, -2.793013467947106e+01, -4.153154110247561e+00, -4.153207117879440e+00, -4.154543536698403e+00, -4.153114077276693e+00, -4.153368284322549e+00, -4.153368284322549e+00, -7.771461241139871e-01, -7.762424860032840e-01, -7.565854639400987e-01, -7.622505740591278e-01, -7.610838302640077e-01, -7.610838302640077e-01, -1.985404504744615e-01, -2.016044752898452e-01, -8.886473086154950e-01, -1.563547260848179e-01, -1.699864611119152e-01, -1.699864611119154e-01, -1.068862867083500e-02, -1.113590555318417e-02, -4.677573903374103e-02, -9.378273934229412e-03, -9.615665336996102e-03, -9.615665336996131e-03, -7.014877295489812e+00, -7.018383582233744e+00, -7.015273919537512e+00, -7.018360259928134e+00, -7.016533590815575e+00, -7.016533590815575e+00, -2.326449432643623e+00, -2.356531891288716e+00, -2.316696162357653e+00, -2.343925871287012e+00, -2.347732720698752e+00, -2.347732720698752e+00, -7.082812933183185e-01, -7.935706830303777e-01, -6.488669866400261e-01, -6.988568607482346e-01, -7.231709001919521e-01, -7.231709001919521e-01, -1.120422220221250e-01, -1.974768088358791e-01, -1.052737035035561e-01, -2.388613832301583e+00, -1.244588970149433e-01, -1.244588970149432e-01, -1.397635784542104e-02, -1.227807309874771e-02, -6.718908661553228e-03, -7.405640148523132e-02, -9.926565220887761e-03, -9.926565220887777e-03, -7.357487235935909e-01, -7.604464358673142e-01, -7.543572453646341e-01, -7.466781138295469e-01, -7.507974362931727e-01, -7.507974362931727e-01, -7.182835051309291e-01, -6.144545466318176e-01, -6.550813550644647e-01, -6.909167721940597e-01, -6.728441897059030e-01, -6.728441897059030e-01, -8.318070391310236e-01, -2.510760760856219e-01, -3.030422853072339e-01, -4.037639331438072e-01, -3.501972615938734e-01, -3.501972615938732e-01, -5.388748751121258e-01, -4.196316015403648e-02, -6.020557228044993e-02, -3.956076411232101e-01, -8.788116412675964e-02, -8.788116412675964e-02, -2.273286126998395e-02, -8.623120034932919e-03, -5.243245025080415e-03, -7.974766556544302e-02, -8.164702007412771e-03, -8.164702007412761e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_tm_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_tm", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-8.193006286225825e-09, -8.192878079005926e-09, -8.190505310631643e-09, -8.192349019975019e-09, -8.191641002198230e-09, -8.191641002198230e-09, -1.155174936716840e-05, -1.155503724556019e-05, -1.164826022440059e-05, -1.167401895735486e-05, -1.164769692488043e-05, -1.164769692488043e-05, -2.507897876988106e-03, -2.510642878124101e-03, -2.676316583522046e-03, -3.225901012495844e-03, -3.039367408912176e-03, -3.039367408912176e-03, -8.336408238903359e-01, -8.193698553229735e-01, 1.336507286540700e-04, -1.122647977480800e+00, -1.080825719107196e+00, -1.080825719107198e+00, -3.833684234770164e+03, -3.403272740471871e+03, -3.218249355724839e+01, -9.657913980363146e+03, -7.353321515452293e+03, -7.353321515452275e+03, -1.164707391640216e-06, -1.154442896748515e-06, -1.161478511093328e-06, -1.152507362279802e-06, -1.160947986968111e-06, -1.160947986968111e-06, -9.658922739977069e-05, -9.277730176307267e-05, -9.384557865815193e-05, -9.025459078848577e-05, -9.591625849018526e-05, -9.591625849018526e-05, -2.537521372355530e-02, -2.202485213666997e-02, -2.426530374450562e-02, -1.542968725998076e-02, -2.576183207065572e-02, -2.576183207065572e-02, -2.547295199004338e+00, -6.043902362056244e-01, -3.016482229500099e+00, -1.171360939068127e-04, -2.389018867516960e+00, -2.389018867516962e+00, 8.119455370865586e+03, -3.536932203151100e+03, -3.987502442239730e+04, -8.636138951667428e+00, -9.505310200186112e+03, -9.505310200186184e+03, -1.969211455539503e-01, -5.910665035723755e-02, -8.492559851163049e-02, -1.215219815951586e-01, -1.009606034589161e-01, -1.009606034589161e-01, -1.183900680216531e-01, -2.391363544701077e-02, -2.529255849418331e-02, -3.557167510254142e-02, -2.904868904542981e-02, -2.904868904542981e-02, -1.381629567849193e-02, -3.094981674234800e-01, -1.970637828734102e-01, -1.194561667728099e-01, -1.541535535084718e-01, -1.541535535084714e-01, -4.604025115707758e-02, -4.157204692367205e+01, -1.493955725032032e+01, -1.746411557142553e-01, -6.034004532229375e+00, -6.034004532229376e+00, -3.008384107080423e+02, 4.402888610371103e+05, -7.568525709772142e+04, -8.135532599770100e+00, -2.031024279610924e+04, -2.031024279610928e+04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_tm_BrOH_cation_restr_1_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_tm", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_tm_BrOH_cation_restr_1_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_tm", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [8.671409438615318e-04, 8.671370997808584e-04, 8.668656813079930e-04, 8.669140799404719e-04, 8.669251227173195e-04, 8.669251227173195e-04, 2.540136557613306e-03, 2.541632636171965e-03, 2.585307137580152e-03, 2.607098066615843e-03, 2.588335051845898e-03, 2.588335051845898e-03, -4.373097060919913e-03, -4.357798527346006e-03, -3.490729317890838e-03, -1.679070706895701e-03, -2.341481371395325e-03, -2.341481371395325e-03, 3.894099787899953e-02, 3.969403535295122e-02, -9.015707884065895e-03, 1.552717110474458e-02, 2.655510704340180e-02, 2.655510704340180e-02, 7.900645739753638e-03, 8.041074244827338e-03, 9.608344511525615e-03, 2.504582143907787e-03, 4.790366711232170e-03, 4.790366711232134e-03, 1.215930379195886e-03, 1.199602088985113e-03, 1.208788661511883e-03, 1.194578276942814e-03, 1.210996463554958e-03, 1.210996463554958e-03, 6.807233616320607e-03, 6.702125098265634e-03, 6.471424409111029e-03, 6.343975237712410e-03, 6.929412973689372e-03, 6.929412973689372e-03, 4.638180067752651e-02, 3.215739546513997e-02, 2.579024103765307e-02, 1.017631659288132e-02, 5.007065741235514e-02, 5.007065741235514e-02, 1.218846714564057e-02, 2.814466879853273e-02, 1.063324299165561e-02, 2.841473066277567e-03, 2.349587362476230e-02, 2.349587362476234e-02, -3.967000495160312e-03, -1.255647373116778e-03, 3.813525499205964e-03, 8.602324895910991e-03, -7.450164488826004e-04, -7.450164488825670e-04, 3.675793429880388e-01, 1.486818565678691e-01, 2.042303426159777e-01, 2.718414726811454e-01, 2.353727663786454e-01, 2.353727663786453e-01, 8.997477590667975e-02, 2.690668578742082e-02, 3.642481258649855e-02, 6.602863414766931e-02, 4.787248825505360e-02, 4.787248825505357e-02, 2.111432949458251e-02, 2.957631856547144e-02, 3.168477136069590e-02, 4.024113825112180e-02, 3.756399981553853e-02, 3.756399981553854e-02, 4.090473532195688e-02, 1.338599757505027e-02, 8.989253577814363e-03, 5.564502262620914e-02, 1.597399087303998e-02, 1.597399087303997e-02, -2.222944708783831e-03, -3.710228551851917e-03, 4.193755028540442e-03, 2.112456583972103e-02, 1.703334750875184e-03, 1.703334750875185e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05