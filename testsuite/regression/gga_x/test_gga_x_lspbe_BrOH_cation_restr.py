
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_lspbe_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lspbe", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.094388843206304e+01, -2.094391595407339e+01, -2.094410588845287e+01, -2.094369502602255e+01, -2.094390069800608e+01, -2.094390069800608e+01, -3.472986146630638e+00, -3.472948343065982e+00, -3.472159673286273e+00, -3.474190607306788e+00, -3.473024540425093e+00, -3.473024540425093e+00, -6.971913287487753e-01, -6.972037754619572e-01, -6.993423778993249e-01, -7.041057373512700e-01, -7.022617213193152e-01, -7.022617213193152e-01, -2.148741165880918e-01, -2.160036421375874e-01, -8.047958279427667e-01, -1.808233438698175e-01, -1.940899258448691e-01, -1.940899258448691e-01, -2.073068987413010e-04, -3.522137775778787e-04, -4.983665708768283e-02, 1.092807903780092e-06, -5.493405507531562e-06, -5.493405507531562e-06, -5.034693765656768e+00, -5.034080094841746e+00, -5.034676263103135e+00, -5.034134346865352e+00, -5.034376736134765e+00, -5.034376736134765e+00, -2.112800264240167e+00, -2.122666071774531e+00, -2.114074666272829e+00, -2.122789935308453e+00, -2.117622833191890e+00, -2.117622833191890e+00, -5.806629725197036e-01, -6.028143642997860e-01, -5.415718905276629e-01, -5.373239258678719e-01, -5.864125708242968e-01, -5.864125708242968e-01, -1.361171980516540e-01, -2.300446600332553e-01, -1.267757286474103e-01, -1.813597699195763e+00, -1.533473827737230e-01, -1.533473827737230e-01, 5.925190046622822e-07, 1.092330186445088e-06, -2.008712227629754e-07, -8.645358808413990e-02, -1.195426152889031e-07, -1.195426152889031e-07, -5.507622471079225e-01, -5.539043346826700e-01, -5.528161258511456e-01, -5.518973592280051e-01, -5.523576411121045e-01, -5.523576411121045e-01, -5.339836202058845e-01, -5.104161824863653e-01, -5.170028558581736e-01, -5.231756246501910e-01, -5.198260598326607e-01, -5.198260598326607e-01, -6.331018670644499e-01, -2.750219923288665e-01, -3.101643307625777e-01, -3.657116354342317e-01, -3.356704723911349e-01, -3.356704723911349e-01, -4.710538285710509e-01, -4.678634182984920e-02, -6.818336434508267e-02, -3.421098227991924e-01, -1.089469094908771e-01, -1.089469094908771e-01, -1.636563532176961e-03, 4.819998932171203e-08, 3.101129014364597e-07, -1.028149213566374e-01, 2.485830621419239e-07, 2.485830621419237e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_lspbe_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lspbe", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.515010456054148e+01, -2.515020397973427e+01, -2.515062065343816e+01, -2.514914202844088e+01, -2.514992039231277e+01, -2.514992039231277e+01, -4.039026942625873e+00, -4.039072539238886e+00, -4.040226620661180e+00, -4.038988391205019e+00, -4.039178977286234e+00, -4.039178977286234e+00, -7.609249837740532e-01, -7.596891346890808e-01, -7.315418898772291e-01, -7.378990586426659e-01, -7.367802270499834e-01, -7.367802270499834e-01, -2.051702115551552e-01, -2.061870350008486e-01, -8.917772596421562e-01, -1.843159759265513e-01, -1.895355138809286e-01, -1.895355138809286e-01, -2.454921392887561e-03, -3.696595698403690e-03, -8.356186565851044e-02, 3.444738244361338e-06, -1.462206954241190e-04, -1.462206954241201e-04, -6.200983665250159e+00, -6.203734956240358e+00, -6.201106930524204e+00, -6.203535807091706e+00, -6.202379844442464e+00, -6.202379844442464e+00, -2.191655339775500e+00, -2.208771139313106e+00, -2.177696380414641e+00, -2.192585138501828e+00, -2.208138054238677e+00, -2.208138054238677e+00, -6.859536070303190e-01, -7.743591319752333e-01, -6.310668473920045e-01, -6.813856972226394e-01, -7.002743085001432e-01, -7.002743085001432e-01, -1.611047870548075e-01, -2.261381023777850e-01, -1.533478388128078e-01, -2.334424178863587e+00, -1.657529115508636e-01, -1.657529115508636e-01, 2.365744142363976e-06, 3.030237921082270e-06, -2.833858826399327e-05, -1.169325687290000e-01, -3.143398676423556e-05, -3.143398676423529e-05, -7.247772298230807e-01, -7.127492937341210e-01, -7.169033415384033e-01, -7.203988803385734e-01, -7.186434365265625e-01, -7.186434365265625e-01, -7.075354688867210e-01, -5.593135998963189e-01, -5.972137612896543e-01, -6.391249611318446e-01, -6.174186443098286e-01, -6.174186443098286e-01, -8.104605018555751e-01, -2.630275293264814e-01, -2.989420360411429e-01, -3.923426864495726e-01, -3.376562001672530e-01, -3.376562001672530e-01, -5.143273350656812e-01, -8.134305841894947e-02, -1.022344686096092e-01, -3.838085546041027e-01, -1.335103064501265e-01, -1.335103064501265e-01, -1.163203172930346e-02, 1.927908318398286e-07, 1.240265162844212e-06, -1.289262324708184e-01, -2.080968149355758e-05, -2.080968149355741e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_lspbe_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lspbe", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-7.026062220449189e-09, -7.026013983275180e-09, -7.025726996189761e-09, -7.026446162326481e-09, -7.026079642821798e-09, -7.026079642821798e-09, -9.468366138185911e-06, -9.468660855760209e-06, -9.474543916699027e-06, -9.457127833557114e-06, -9.467806536173932e-06, -9.467806536173932e-06, -6.015841728724524e-03, -6.019469657053617e-03, -6.026417255512224e-03, -5.862398317958383e-03, -5.922610525250054e-03, -5.922610525250054e-03, -6.031164469424376e-01, -6.029742448269698e-01, -3.365859753981943e-03, -7.703463305115198e-01, -7.520595409344402e-01, -7.520595409344397e-01, 3.219109259217695e+02, 4.204017639470551e+02, 1.026122753032663e+01, -1.452138878479982e+00, 5.766568034397471e+01, 5.766568034397523e+01, -2.070861454267789e-06, -2.071117707329539e-06, -2.070859148088857e-06, -2.071085596427912e-06, -2.070999297258895e-06, -2.070999297258895e-06, -7.246557932985781e-05, -7.108584619491149e-05, -7.236832129521769e-05, -7.116086247635179e-05, -7.173457686530894e-05, -7.173457686530894e-05, -1.200871735488035e-02, -9.797487792523403e-03, -1.599513814285349e-02, -1.565790606243688e-02, -1.147077470078830e-02, -1.147077470078830e-02, -6.867111099320928e-01, -3.658332880914270e-01, -6.839264664940757e-01, -1.194199527835490e-04, -1.004687340127439e+00, -1.004687340127439e+00, -2.876485448802429e+00, -1.335140728889066e+00, 1.335122660667753e+02, 2.641587968275949e-01, 5.803946766160931e+01, 5.803946766160875e+01, -1.382388640794336e-02, -1.372765484942472e-02, -1.376136583491323e-02, -1.378979385072568e-02, -1.377563003353660e-02, -1.377563003353660e-02, -1.556815844863397e-02, -2.090628310551076e-02, -1.935272476785995e-02, -1.785326487509293e-02, -1.863693607345971e-02, -1.863693607345971e-02, -8.072125494814716e-03, -2.186048005484633e-01, -1.510098319357811e-01, -7.998599930884782e-02, -1.138911635866044e-01, -1.138911635866044e-01, -2.886343700344226e-02, 1.232992902394522e+01, 2.978223648229886e+00, -1.024792150484615e-01, -1.027375303573835e+00, -1.027375303573837e+00, 4.464194134039953e+02, -1.264153320826783e+01, -6.118045741805791e+00, -8.482051359140880e-01, 5.565455412087783e+01, 5.565455412087757e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05