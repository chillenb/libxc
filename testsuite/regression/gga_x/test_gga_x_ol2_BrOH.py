
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_ol2_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ol2", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.120579230227179e+01, -2.120581493734149e+01, -2.120599552473483e+01, -2.120562324244135e+01, -2.120580357383736e+01, -2.120580357383736e+01, -3.513242197675079e+00, -3.513217499502927e+00, -3.512682741830462e+00, -3.514348175097485e+00, -3.513244460931843e+00, -3.513244460931843e+00, -6.987061383921978e-01, -6.985583114566182e-01, -6.961886341788044e-01, -7.007334980974763e-01, -6.986494514852228e-01, -6.986494514852228e-01, -2.119856569534169e-01, -2.128928379439607e-01, -8.192259209583639e-01, -1.862679847784079e-01, -2.122271916018661e-01, -2.122271916018661e-01, -4.983086300695374e-01, -4.758257461128296e-01, -2.111890963792002e-01, -1.218650611828109e+00, -4.717483520512256e-01, -4.717483520512256e-01, -5.081823296308865e+00, -5.080655004078576e+00, -5.081712121958097e+00, -5.080803699966642e+00, -5.081214849773918e+00, -5.081214849773918e+00, -2.097371604353757e+00, -2.108193038215896e+00, -2.096989458322482e+00, -2.105402212261932e+00, -2.105246469457050e+00, -2.105246469457050e+00, -5.810730959921584e-01, -5.856770148768280e-01, -5.533941156646901e-01, -5.475858949994231e-01, -5.989517293261425e-01, -5.989517293261425e-01, -1.822103730415137e-01, -2.360483339389058e-01, -1.755012520558412e-01, -1.782259171951873e+00, -1.761307533945527e-01, -1.761307533945527e-01, -1.226339220733729e+00, -1.072931659693361e+00, -8.459888275258672e-01, -1.754911552256288e-01, -9.689406611797752e-01, -9.689406611797753e-01, -5.058457187626118e-01, -5.488865688408789e-01, -5.397884124165184e-01, -5.294481201252454e-01, -5.351117693051592e-01, -5.351117693051594e-01, -4.696692887036467e-01, -5.196562828632006e-01, -5.295153080423831e-01, -5.348484724967597e-01, -5.326721746760003e-01, -5.326721746760003e-01, -6.174154490769610e-01, -2.759022321900628e-01, -3.090046817575931e-01, -3.671889428444655e-01, -3.355300970627146e-01, -3.355300970627145e-01, -4.709327625455282e-01, -2.396749622852122e-01, -1.962476351602897e-01, -3.393746274542350e-01, -1.618907963368831e-01, -1.618907963368831e-01, -4.919993941747843e-01, -2.977758235910794e+00, -1.690820932995022e+00, -1.536405764826312e-01, -9.429588427629434e-01, -9.429588427629457e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_ol2_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ol2", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.529794392406377e+01, -2.529797766777936e+01, -2.529821342295847e+01, -2.529765905065940e+01, -2.529796100712840e+01, -2.529796100712840e+01, -4.164068572615211e+00, -4.164060161432283e+00, -4.163997332580299e+00, -4.164984528291760e+00, -4.164081985972962e+00, -4.164081985972962e+00, -8.013902203158719e-01, -8.004348470228041e-01, -7.725117649654367e-01, -7.791039615172149e-01, -8.010432784496475e-01, -8.010432784496475e-01, -2.038129389614791e-01, -2.084570730598864e-01, -9.609880442165110e-01, -1.094459450650196e-01, -2.052426896108036e-01, -2.052426896108036e-01, 6.362360470410479e-01, 6.049249735131658e-01, 1.622912855832907e-01, 1.611334186172022e+00, 5.997959012159725e-01, 5.997959012159725e-01, -6.066334723784381e+00, -6.064656321076669e+00, -6.066174482588463e+00, -6.064869573687687e+00, -6.065464265474283e+00, -6.065464265474283e+00, -2.331257389692979e+00, -2.349777648787528e+00, -2.321517926072060e+00, -2.336087168409204e+00, -2.356037874519308e+00, -2.356037874519308e+00, -6.915527200556280e-01, -6.842784352838651e-01, -6.576422423560720e-01, -6.491515626977448e-01, -7.143717369648271e-01, -7.143717369648271e-01, -2.199089924806682e-02, -1.799001513791017e-01, -2.742246612781979e-02, -2.082950585832572e+00, -7.027977759118724e-02, -7.027977759118724e-02, 1.622063302553789e+00, 1.415657079030849e+00, 1.116803333373992e+00, 6.824067908490757e-02, 1.278327910490735e+00, 1.278327910490735e+00, -5.469901561789089e-01, -6.377182423249119e-01, -6.184837658064408e-01, -5.965060082776805e-01, -6.085424509850363e-01, -6.085424509850365e-01, -4.893941561232262e-01, -6.052684034380739e-01, -6.282974603205688e-01, -6.386386433954131e-01, -6.351716981795626e-01, -6.351716981795626e-01, -7.238795978253211e-01, -2.503853615753315e-01, -3.162231505685784e-01, -4.217730369788019e-01, -3.683600848481719e-01, -3.683600848481720e-01, -5.458733166940659e-01, 2.089786484781075e-01, 1.123420732207412e-01, -3.985393354160966e-01, 4.996647005399866e-04, 4.996647005399403e-04, 6.229844011255412e-01, 3.966256873954456e+00, 2.246665438254345e+00, -5.829458111480340e-04, 1.245725408029778e+00, 1.245725408029781e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_ol2_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ol2", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-7.536411007750451e-09, -7.536505236195873e-09, -7.536629677141037e-09, -7.535093069448444e-09, -7.536463173820532e-09, -7.536463173820532e-09, -8.326515381354433e-06, -8.327404427637474e-06, -8.350579234055370e-06, -8.303740474651283e-06, -8.326830717304180e-06, -8.326830717304180e-06, -4.429018660343649e-03, -4.425958373088626e-03, -4.428149997241066e-03, -4.311741801462521e-03, -4.427962494767864e-03, -4.427962494767864e-03, -6.137192791486037e-01, -5.867893878826832e-01, -2.536493181458722e-03, -2.088411912968574e+00, -6.054852989213312e-01, -6.054852989213312e-01, -5.807099669354944e+04, -4.824156239467041e+04, -1.768709383893182e+02, -1.096366797655867e+06, -5.030596365078235e+04, -5.030596365078235e+04, -2.873841809202979e-06, -2.894227153662331e-06, -2.875852554609183e-06, -2.891701104450830e-06, -2.884335952799351e-06, -2.884335952799351e-06, -5.372772570095406e-05, -5.259846151239946e-05, -5.384194529874477e-05, -5.294171123173794e-05, -5.286940622182026e-05, -5.286940622182026e-05, -1.212743953191094e-02, -3.649576612999995e-02, -1.420494456474858e-02, -3.130407170271805e-02, -1.168435985582257e-02, -1.168435985582257e-02, -7.064829701401122e+00, -5.693417935797996e-01, -7.443089339850692e+00, -4.224587055371528e-04, -3.975803171763621e+00, -3.975803171763621e+00, -1.265931311618583e+06, -7.421194217680638e+05, -2.351664671783030e+06, -4.689883786238575e+01, -1.076724670895584e+06, -1.076724670895584e+06, -2.158768044815984e-01, -5.373764138550044e-02, -7.552462539425871e-02, -1.074353345108308e-01, -8.894295132270291e-02, -8.894295132270282e-02, -4.105269941111579e-01, -1.509235981789727e-02, -1.643503710886873e-02, -2.232216123267274e-02, -1.844974010007517e-02, -1.844974010007523e-02, -2.709924040605107e-02, -2.339076323683564e-01, -1.239742212203796e-01, -5.821230570508960e-02, -8.250475321456288e-02, -8.250475321456284e-02, -2.198816548374999e-02, -2.403184197528170e+02, -7.160595964098725e+01, -8.677483916423605e-02, -1.625214595211126e+01, -1.625214595211126e+01, -3.077957308785706e+04, -1.320835983352168e+08, -1.014302956506404e+07, -1.962059507833758e+01, -2.063950572522310e+06, -2.063950572522322e+06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05