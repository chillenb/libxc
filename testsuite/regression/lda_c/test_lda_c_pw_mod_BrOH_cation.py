
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_pw_mod_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_pw_mod", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.645995504598874e-01, -1.645996352769430e-01, -1.646000385289871e-01, -1.645987760591848e-01, -1.645994338376557e-01, -1.645994338376557e-01, -1.107858338283929e-01, -1.107858953703322e-01, -1.107880165538513e-01, -1.107897430660369e-01, -1.107865840356207e-01, -1.107865840356207e-01, -6.701332647009473e-02, -6.698625374487631e-02, -6.633550766928430e-02, -6.653684597800753e-02, -6.628241523783371e-02, -6.628241523783371e-02, -3.784104461181498e-02, -3.810773248299326e-02, -7.071248082080384e-02, -3.294245817029351e-02, -2.771143571516880e-02, -2.771143571516881e-02, -3.773696785719675e-03, -3.937471296854703e-03, -1.472968080447034e-02, -2.355986619441690e-03, -2.448489658576850e-03, -2.448489658576850e-03, -1.224262466632711e-01, -1.224315961438950e-01, -1.224265129464943e-01, -1.224312356359921e-01, -1.224289475121666e-01, -1.224289475121666e-01, -9.471366583147260e-02, -9.491133084943865e-02, -9.456832610819237e-02, -9.474390937311164e-02, -9.489144390379985e-02, -9.489144390379985e-02, -6.380950346837234e-02, -6.578529679472132e-02, -6.192000199282998e-02, -6.284039868666000e-02, -6.406349219536719e-02, -6.406349219536717e-02, -2.699235473841160e-02, -3.791208560257751e-02, -2.575800141222741e-02, -9.418618612322655e-02, -2.944233340534604e-02, -2.944233340534604e-02, -1.871459774611198e-03, -2.309612356171233e-03, -1.817197404218478e-03, -2.031081282349249e-02, -2.023384882378595e-03, -2.023384882378594e-03, -6.392396594984479e-02, -6.378756852459780e-02, -6.383563580100113e-02, -6.387517814757501e-02, -6.385539257642651e-02, -6.385539257642651e-02, -6.327375432049363e-02, -5.971164787643432e-02, -6.077755274889974e-02, -6.179616844032595e-02, -6.128070638566306e-02, -6.128070638566306e-02, -6.692281220054765e-02, -4.238586195740855e-02, -4.605979358136188e-02, -5.192725174127702e-02, -4.894363551560387e-02, -4.894363551560387e-02, -5.781814550942019e-02, -1.432337785401933e-02, -1.764153240911863e-02, -5.112859097285417e-02, -2.350608846529213e-02, -2.350608846529213e-02, -5.037446550363285e-03, -6.963063398225441e-04, -1.374982125405463e-03, -2.268013163367492e-02, -1.910123256188000e-03, -1.910123256188002e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_pw_mod_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_pw_mod", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.747369753456869e-01, -1.747373592460347e-01, -1.747369289286906e-01, -1.747375762850792e-01, -1.747380160558852e-01, -1.747373003530871e-01, -1.747352529548414e-01, -1.747375238520149e-01, -1.747351501822271e-01, -1.747389498785631e-01, -1.747351501822271e-01, -1.747389498785631e-01, -1.202893499889770e-01, -1.202922014327670e-01, -1.202892104901381e-01, -1.202924665416908e-01, -1.202951542125395e-01, -1.202908517166852e-01, -1.202962311684931e-01, -1.202932980984655e-01, -1.203306534397094e-01, -1.202524715692195e-01, -1.203306534397094e-01, -1.202524715692195e-01, -7.534022876474468e-02, -7.481054072997755e-02, -7.537293704871807e-02, -7.472260167139465e-02, -7.390882988754914e-02, -7.482611619929942e-02, -7.471850704775328e-02, -7.443058546683960e-02, -7.037457509512338e-02, -7.906556863981584e-02, -7.037457509512338e-02, -7.906556863981584e-02, -4.564060327237250e-02, -4.259099551694412e-02, -4.617672169257560e-02, -4.268429759163285e-02, -8.138131009521569e-02, -7.673667638414071e-02, -3.918020935136773e-02, -3.821967190018716e-02, -2.640799090440606e-02, -7.149919794390105e-02, -2.640799090440609e-02, -7.149919794390104e-02, -5.065681489236365e-03, -4.646401610809158e-03, -5.316900325349619e-03, -4.819083847930042e-03, -1.901496399271941e-02, -1.733574795173590e-02, -3.011597533973102e-03, -3.081336537258555e-03, -2.735674303888032e-03, -5.360986007450113e-03, -2.735674303888032e-03, -5.360986007450113e-03, -1.321131074354259e-01, -1.321678349608994e-01, -1.321178186608825e-01, -1.321739921705766e-01, -1.321128689650321e-01, -1.321686151017558e-01, -1.321180840697961e-01, -1.321729934978753e-01, -1.321153491000285e-01, -1.321710807519585e-01, -1.321153491000285e-01, -1.321710807519585e-01, -1.038230376850377e-01, -1.038322271691791e-01, -1.040061653898525e-01, -1.040557531149889e-01, -1.038354328794433e-01, -1.035216009444120e-01, -1.040229030307593e-01, -1.036954400908969e-01, -1.036133129641505e-01, -1.044127911216831e-01, -1.036133129641505e-01, -1.044127911216831e-01, -7.154035719371461e-02, -7.188021372637021e-02, -7.381532476082853e-02, -7.375554194115297e-02, -7.209384314263843e-02, -6.758533476991320e-02, -7.285616976905461e-02, -6.872088296726787e-02, -6.892990125395577e-02, -7.551251921943604e-02, -6.892990125395576e-02, -7.551251921943600e-02, -3.239197726005828e-02, -3.184391527087424e-02, -4.430920677707256e-02, -4.391091852170306e-02, -3.228823617617899e-02, -2.939977194247364e-02, -1.032443250912971e-01, -1.033256406767505e-01, -3.680158193982921e-02, -3.317141602912393e-02, -3.680158193982921e-02, -3.317141602912393e-02, -2.493594694998078e-03, -2.368514638763016e-03, -3.017775709737398e-03, -2.956742417356703e-03, -2.456095252278784e-03, -2.275318542350691e-03, -2.470723487135053e-02, -2.444200035936174e-02, -3.448524464083225e-03, -2.295778297885341e-03, -3.448524464083225e-03, -2.295778297885340e-03, -7.207907563633678e-02, -7.158366737767322e-02, -7.193770326557357e-02, -7.143826062122849e-02, -7.198832468635549e-02, -7.148872356177728e-02, -7.202773238186704e-02, -7.153241948061712e-02, -7.200798844943504e-02, -7.151058106231506e-02, -7.200798844943504e-02, -7.151058106231506e-02, -7.135713929633580e-02, -7.093720694789084e-02, -6.765589239010464e-02, -6.713345832660030e-02, -6.879092177868548e-02, -6.824727782565454e-02, -6.981057386457169e-02, -6.937321679273566e-02, -6.927821647882314e-02, -6.881965680321915e-02, -6.927821647882314e-02, -6.881965680321915e-02, -7.508806671297258e-02, -7.487032192240953e-02, -4.920038482173786e-02, -4.868825129823109e-02, -5.337095172613578e-02, -5.241584993200162e-02, -5.944371803360495e-02, -5.885560748526495e-02, -5.595126563767337e-02, -5.598694290621807e-02, -5.595126563767337e-02, -5.598694290621806e-02, -6.580000808900417e-02, -6.499414302323578e-02, -1.773472996354029e-02, -1.753120798266075e-02, -2.214692961715042e-02, -2.091744112127225e-02, -5.910188411883857e-02, -5.752755528065093e-02, -2.967981761157849e-02, -2.694165139683765e-02, -2.967981761157849e-02, -2.694165139683765e-02, -6.593851617928385e-03, -6.257235187311729e-03, -9.143293617831678e-04, -9.116957734166457e-04, -1.869567052566781e-03, -1.724808884064574e-03, -2.767194528635372e-02, -2.688551334493897e-02, -3.151293236449766e-03, -2.184548869499210e-03, -3.151293236449767e-03, -2.184548869499210e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05